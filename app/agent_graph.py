"""
LangGraph orchestration for multi-agent query processing.

This module defines the agent graph workflow that routes queries between agents:
1. RAG Search Agent - searches documentation directly
2. Medicine Name Extraction Agent - extracts medicine names and searches for them
3. Full Instruction Agent - retrieves full instructions and generates comprehensive answers

Uses LangGraph for intelligent routing between agents based on query results.
"""

import logging
from typing import Literal, TypedDict

import chromadb
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agents import (
    INSUFFICIENT_INFO_MARKER,
    create_full_instruction_agent,
    create_medicine_extraction_agent,
    create_rag_search_agent,
    extract_medicine_ids,
)
from app.llm_providers import get_model_name

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between agents in the graph."""
    query: str
    answer: str
    medicine_ids: list[str]
    agent_used: str
    response_language: str
    error: str
    sources: list[dict]  # List of source dictionaries with file paths, medicine_ids, etc.
    tools_used: list[str]  # List of tool names used by agents


def should_continue_to_medicine_search(state: AgentState) -> Literal["extract_medicines", "end"]:
    """
    Decide whether to continue to medicine extraction based on RAG search results.
    
    Checks for the special marker that the agent returns when insufficient information is found.
    This works regardless of the response language.
    
    Returns:
        "extract_medicines" if answer contains insufficient info marker, "end" if sufficient
    """
    answer = state.get("answer", "")
    logger.debug(f"🔀 Routing decision: Checking answer ({len(answer)} chars)")
    
    # Check if the answer contains the insufficient info marker
    # This marker is language-independent and explicitly returned by the agent
    has_insufficient_marker = INSUFFICIENT_INFO_MARKER in answer
    
    # If marker is present, continue to medicine extraction
    if has_insufficient_marker:
        logger.info(f"➡️ Routing to medicine extraction: Found insufficient info marker")
        return "extract_medicines"
    
    # Also check if answer is too short (less than 30 characters) as a fallback
    # This catches cases where agent might not have followed instructions
    if len(answer.strip()) < 30:
        logger.info(f"➡️ Routing to medicine extraction: Answer too short ({len(answer.strip())} chars)")
        return "extract_medicines"
    
    # Otherwise, the answer is sufficient
    logger.info(f"✅ Routing to end: Answer sufficient ({len(answer.strip())} chars)")
    return "end"


def create_agent_graph(
    llm: BaseChatModel,
    rag_collection: chromadb.Collection,
    medicines_collection: chromadb.Collection,
    embedding_model,
    response_language: str = "English",
):
    """
    Create LangGraph workflow for multi-agent query processing.
    
    Flow:
    1. RAG Search -> Check if sufficient
    2. If insufficient -> Medicine Extraction -> Full Instruction
    3. Return final answer
    """
    
    # Create agents
    rag_agent = create_rag_search_agent(llm, rag_collection, medicines_collection, embedding_model)
    medicine_agent = create_medicine_extraction_agent(llm, medicines_collection, embedding_model)
    full_instruction_agent = create_full_instruction_agent(
        llm, rag_collection, medicines_collection, embedding_model, response_language
    )
    
    # Define node functions
    def rag_search_node(state: AgentState) -> AgentState:
        """Node 1: RAG Search"""
        query = state.get("query", "")
        logger.info(f"🔍 RAG Search Agent: Processing query: '{query[:100]}...'")
        logger.info("   Note: Agent may call tools multiple times to refine search - this is normal behavior")
        try:
            logger.debug(f"Invoking RAG agent with query: {query}")
            result = rag_agent.invoke({"messages": [{"role": "user", "content": query}]})
            logger.debug(f"RAG agent returned result with {len(result.get('messages', []))} messages")
            
            # Extract answer, sources, and tools from messages
            messages = result.get("messages", [])
            sources = state.get("sources", [])
            tools_used = state.get("tools_used", [])
            
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    answer = last_message.content
                    # Extract tool calls from AI message
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                            if tool_name:
                                if tool_name not in tools_used:
                                    tools_used.append(tool_name)
                                # Log tool call with parameters
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  🔧 Tool call: {tool_name}({args_str})")
                else:
                    answer = str(last_message.get("content", "")) if isinstance(last_message, dict) else str(last_message)
                logger.info(f"✅ RAG Search Agent: Got answer ({len(answer)} chars)")
                logger.debug(f"Answer preview: {answer[:200]}...")
                
                # Extract sources and tool names from tool messages
                import json
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        # Extract tool name
                        tool_name = getattr(msg, "name", None)
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_name:
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                            # Try to find corresponding tool call to get parameters
                            tool_args = {}
                            for prev_msg in messages:
                                if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, "tool_calls"):
                                    for tc in prev_msg.tool_calls:
                                        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                                        if tc_id == tool_call_id:
                                            tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                            break
                            if tool_args:
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  ✅ Tool result: {tool_name}({args_str})")
                            else:
                                logger.debug(f"  ✅ Tool result: {tool_name}")
                        
                        try:
                            # Try to parse tool result as JSON to extract source info
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                            if isinstance(tool_result, list):
                                for item in tool_result:
                                    if isinstance(item, dict) and "source_file" in item:
                                        source_info = {
                                            "medicine_id": item.get("medicine_id", "N/A"),
                                            "source_file": item.get("source_file", "N/A"),
                                            "chunk_id": item.get("chunk_id", "N/A"),
                                            "chunk_index": item.get("chunk_index", "N/A"),
                                            "chunk_text": item.get("document", "")[:500] if item.get("document") else "",
                                        }
                                        # Avoid duplicates
                                        if source_info not in sources:
                                            sources.append(source_info)
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, try to extract file names from text
                            pass
                
                if tools_used:
                    logger.info(f"  Tools used: {', '.join(tools_used)}")
            else:
                answer = ""
                tools_used = state.get("tools_used", [])
                logger.warning("⚠️ RAG Search Agent: No messages in result")
            
            return {
                **state,
                "answer": answer,
                "sources": sources,
                "tools_used": tools_used,
                "agent_used": "rag_search",
            }
        except Exception as e:
            logger.error(f"❌ RAG Search Agent error: {e}", exc_info=True)
            return {
                **state,
                "error": f"RAG search error: {e}",
                "agent_used": "rag_search",
            }
    
    def extract_medicines_node(state: AgentState) -> AgentState:
        """Node 2: Extract Medicine Names"""
        query = state.get("query", "")
        logger.info(f"💊 Medicine Extraction Agent: Extracting medicine names from query: '{query[:100]}...'")
        try:
            extraction_prompt = f"""The user asked: "{query}"

Extract medicine names from this query and search for them in the database. 
Return ONLY the medicine IDs you find (32-character hex strings), one per line or comma-separated."""
            
            logger.debug(f"Invoking medicine extraction agent")
            result = medicine_agent.invoke({"messages": [{"role": "user", "content": extraction_prompt}]})
            logger.debug(f"Medicine extraction agent returned result with {len(result.get('messages', []))} messages")
            
            # Extract answer, sources, and tools from messages
            messages = result.get("messages", [])
            sources = state.get("sources", [])
            tools_used = state.get("tools_used", [])
            
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    medicine_ids_text = last_message.content
                    # Extract tool calls from AI message
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                            if tool_name:
                                if tool_name not in tools_used:
                                    tools_used.append(tool_name)
                                # Log tool call with parameters
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  🔧 Tool call: {tool_name}({args_str})")
                else:
                    medicine_ids_text = str(last_message.get("content", "")) if isinstance(last_message, dict) else str(last_message)
                logger.debug(f"Medicine extraction response: {medicine_ids_text[:200]}...")
                
                # Extract sources and tool names from tool messages (medicine search results)
                import json
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        # Extract tool name
                        tool_name = getattr(msg, "name", None)
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_name:
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                            # Try to find corresponding tool call to get parameters
                            tool_args = {}
                            for prev_msg in messages:
                                if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, "tool_calls"):
                                    for tc in prev_msg.tool_calls:
                                        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                                        if tc_id == tool_call_id:
                                            tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                            break
                            if tool_args:
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  ✅ Tool result: {tool_name}({args_str})")
                            else:
                                logger.debug(f"  ✅ Tool result: {tool_name}")
                        
                        try:
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                            if isinstance(tool_result, list):
                                for item in tool_result:
                                    if isinstance(item, dict) and "medicine_id" in item:
                                        # Medicine search result - we'll get full sources in next step
                                        pass
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                if tools_used:
                    logger.info(f"  Tools used: {', '.join(tools_used)}")
            else:
                medicine_ids_text = ""
                tools_used = state.get("tools_used", [])
                logger.warning("⚠️ Medicine Extraction Agent: No messages in result")
            
            medicine_ids = extract_medicine_ids(medicine_ids_text)
            logger.info(f"✅ Medicine Extraction Agent: Found {len(medicine_ids)} medicine IDs: {medicine_ids}")
            
            return {
                **state,
                "medicine_ids": medicine_ids,
                "sources": sources,
                "tools_used": tools_used,
                "agent_used": "medicine_extraction",
            }
        except Exception as e:
            logger.error(f"❌ Medicine Extraction Agent error: {e}", exc_info=True)
            return {
                **state,
                "error": f"Medicine extraction error: {e}",
                "agent_used": "medicine_extraction",
            }
    
    def full_instruction_node(state: AgentState) -> AgentState:
        """Node 3: Get Full Instructions"""
        query = state.get("query", "")
        medicine_ids = state.get("medicine_ids", [])
        logger.info(f"📚 Full Instruction Agent: Processing {len(medicine_ids)} medicine IDs for query: '{query[:100]}...'")
        
        try:
            if not medicine_ids:
                logger.warning("⚠️ Full Instruction Agent: No medicine IDs provided")
                return {
                    **state,
                    "answer": state.get("answer", "Could not find medicine IDs."),
                    "agent_used": "full_instruction",
                }
            
            instruction_prompt = f"""The user asked: "{query}"

I found these medicine IDs: {', '.join(medicine_ids)}

Please retrieve the full instructions for these medicines using get_medicine_full_instruction 
and provide a comprehensive answer to the user's question in {state.get('response_language', 'English')}."""
            
            logger.debug(f"Invoking full instruction agent with {len(medicine_ids)} medicine IDs")
            result = full_instruction_agent.invoke({"messages": [{"role": "user", "content": instruction_prompt}]})
            logger.debug(f"Full instruction agent returned result with {len(result.get('messages', []))} messages")
            
            # Extract answer, sources, and tools from messages
            messages = result.get("messages", [])
            sources = state.get("sources", [])
            tools_used = state.get("tools_used", [])
            
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    answer = last_message.content
                    # Extract tool calls from AI message
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                            if tool_name:
                                if tool_name not in tools_used:
                                    tools_used.append(tool_name)
                                # Log tool call with parameters
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  🔧 Tool call: {tool_name}({args_str})")
                else:
                    answer = str(last_message.get("content", "")) if isinstance(last_message, dict) else str(last_message)
                logger.info(f"✅ Full Instruction Agent: Got answer ({len(answer)} chars)")
                logger.debug(f"Answer preview: {answer[:200]}...")
                
                # Extract sources and tool names from tool messages (full instruction chunks)
                import json
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        # Extract tool name
                        tool_name = getattr(msg, "name", None)
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_name:
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                            # Try to find corresponding tool call to get parameters
                            tool_args = {}
                            for prev_msg in messages:
                                if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, "tool_calls"):
                                    for tc in prev_msg.tool_calls:
                                        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                                        if tc_id == tool_call_id:
                                            tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                            break
                            if tool_args:
                                args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                                logger.info(f"  ✅ Tool result: {tool_name}({args_str})")
                            else:
                                logger.debug(f"  ✅ Tool result: {tool_name}")
                        
                        try:
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                            if isinstance(tool_result, list):
                                for item in tool_result:
                                    if isinstance(item, dict) and "source_file" in item:
                                        source_info = {
                                            "medicine_id": item.get("medicine_id", medicine_ids[0] if medicine_ids else "N/A"),
                                            "source_file": item.get("source_file", "N/A"),
                                            "chunk_id": item.get("chunk_id", "N/A"),
                                            "chunk_index": item.get("chunk_index", "N/A"),
                                            "chunk_text": item.get("document", "")[:500] if item.get("document") else "",
                                        }
                                        # Avoid duplicates
                                        if source_info not in sources:
                                            sources.append(source_info)
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                if tools_used:
                    logger.info(f"  Tools used: {', '.join(tools_used)}")
            else:
                answer = ""
                tools_used = state.get("tools_used", [])
                logger.warning("⚠️ Full Instruction Agent: No messages in result")
            
            return {
                **state,
                "answer": answer,
                "sources": sources,
                "tools_used": tools_used,
                "agent_used": "full_instruction",
            }
        except Exception as e:
            logger.error(f"❌ Full Instruction Agent error: {e}", exc_info=True)
            return {
                **state,
                "error": f"Full instruction error: {e}",
                "agent_used": "full_instruction",
            }
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("extract_medicines", extract_medicines_node)
    workflow.add_node("full_instruction", full_instruction_node)
    
    # Set entry point
    workflow.set_entry_point("rag_search")
    
    # Add conditional edge: after RAG search, decide whether to continue
    workflow.add_conditional_edges(
        "rag_search",
        should_continue_to_medicine_search,
        {
            "extract_medicines": "extract_medicines",
            "end": END,
        }
    )
    
    # After medicine extraction, always go to full instruction
    workflow.add_edge("extract_medicines", "full_instruction")
    
    # After full instruction, end
    workflow.add_edge("full_instruction", END)
    
    return workflow.compile()


def process_query_with_agents(
    query: str,
    llm: BaseChatModel,
    rag_collection: chromadb.Collection,
    medicines_collection: chromadb.Collection,
    embedding_model,
    response_language: str = "English",
) -> dict:
    """
    Process a query using the LangGraph multi-agent system.
    
    Flow:
    1. Agent 1 tries to answer using RAG search
    2. If Agent 1 cannot find enough information, Agent 2 extracts medicine names
    3. Agent 3 retrieves full instructions and generates comprehensive answer
    
    Returns:
        dict with 'answer', 'sources', 'agent_used', 'error'
    """
    import time
    start_time = time.time()
    logger.info(f"🚀 Starting query processing: '{query[:100]}...'")
    logger.info(f"   Response language: {response_language}")
    logger.info(f"   Model: {get_model_name(llm)}")
    
    try:
        # Create graph
        logger.debug("Creating agent graph...")
        graph = create_agent_graph(
            llm, rag_collection, medicines_collection, embedding_model, response_language
        )
        logger.debug("Agent graph created successfully")
        
        # Initial state
        initial_state: AgentState = {
            "query": query,
            "answer": "",
            "medicine_ids": [],
            "agent_used": "",
            "response_language": response_language,
            "error": "",
            "sources": [],
            "tools_used": [],
        }
        
        # Run graph
        logger.info("Invoking agent graph...")
        graph_start = time.time()
        final_state = graph.invoke(initial_state)
        graph_time = time.time() - graph_start
        logger.info(f"✅ Agent graph completed in {graph_time:.2f}s")
        
        # Remove the marker from final answer if present (it's only used for routing)
        final_answer = final_state.get("answer", "")
        if INSUFFICIENT_INFO_MARKER in final_answer:
            final_answer = final_answer.replace(INSUFFICIENT_INFO_MARKER, "").strip()
            logger.debug("Removed insufficient info marker from final answer")
        
        agent_used = final_state.get("agent_used", "unknown")
        error = final_state.get("error")
        tools_used = final_state.get("tools_used", [])
        total_time = time.time() - start_time
        
        logger.info(f"✅ Query processing completed in {total_time:.2f}s")
        logger.info(f"   Agent used: {agent_used}")
        logger.info(f"   Answer length: {len(final_answer)} chars")
        if tools_used:
            logger.info(f"   Tools used: {', '.join(tools_used)}")
        if error:
            logger.error(f"   Error: {error}")
        if final_state.get("medicine_ids"):
            logger.info(f"   Medicine IDs found: {final_state.get('medicine_ids')}")
        
        result = {
            "answer": final_answer,
            "sources": final_state.get("sources", []),
            "agent_used": agent_used,
            "medicine_ids": final_state.get("medicine_ids", []),
            "tools_used": tools_used,
            "model": get_model_name(llm),
        }
        # Only include error if it's not empty
        if error:
            result["error"] = error
        
        logger.info(f"   Sources found: {len(result['sources'])}")
        return result
        
    except Exception as e:
        import traceback
        total_time = time.time() - start_time
        logger.error(f"❌ Query processing failed after {total_time:.2f}s: {e}", exc_info=True)
        return {
            "error": f"Error processing query: {e}",
            "agent_used": "error",
            "traceback": traceback.format_exc(),
        }