"""
Agent definitions for medical instruction queries.

This module contains factory functions for creating specialized agents:
1. RAG Search Agent - searches documentation directly
2. Medicine Name Extraction Agent - extracts medicine names and searches for them
3. Full Instruction Agent - retrieves full instructions and generates comprehensive answers
"""

import json
import re

import chromadb

# Check LangChain version
try:
    import langchain
    langchain_version = langchain.__version__
    version_parts = [int(x) for x in langchain_version.split('.')[:2]]
    if version_parts < [1, 0]:
        raise ImportError(
            f"LangChain version {langchain_version} is too old. "
            "Please upgrade to LangChain >= 1.0.0. "
            "Make sure you're using the virtual environment: source .venv/bin/activate"
        )
except ImportError:
    raise ImportError(
        "LangChain is not installed. "
        "Please install dependencies: pip install -r requirements.txt"
    )

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from app.rag_tools import create_rag_tools

# Special marker that agent should return when insufficient information is found
# This marker should be included in the answer when the agent cannot find enough information
INSUFFICIENT_INFO_MARKER = "[[INSUFFICIENT_INFO]]"


def create_rag_search_agent(
    llm: BaseChatModel,
    rag_collection: chromadb.Collection,
    medicines_collection: chromadb.Collection,
    embedding_model,
):
    """
    Create Agent 1: RAG Search Agent
    
    This agent searches the medical instruction database directly.
    Uses only search_medical_information tool for focused semantic search.
    Returns a CompiledStateGraph (LangGraph) that can be invoked with messages.
    """
    tools_dict = create_rag_tools(rag_collection, medicines_collection, embedding_model)
    # Use only search_medical_information tool
    tools = [tools_dict["search_medical_information"]]
    
    system_prompt = f"""You are a medical information assistant. Your task is to search medical information 
and answer questions based on the documentation.

You have access to this tool:
- search_medical_information: Search the medical instruction database using semantic search

Your workflow:
1. Use search_medical_information to find relevant information about the user's question
2. If you find relevant information, provide a clear answer citing your sources
3. If you cannot find enough information to answer the question, you MUST include the exact marker "{INSUFFICIENT_INFO_MARKER}" in your response (at the beginning or end)

IMPORTANT: Only use the marker "{INSUFFICIENT_INFO_MARKER}" when you truly cannot find sufficient information. If you found some information but it's incomplete, still provide what you found without the marker.

Always cite your sources when providing information."""
    
    return create_agent(llm, tools=tools, system_prompt=system_prompt)


def create_medicine_extraction_agent(
    llm: BaseChatModel,
    medicines_collection: chromadb.Collection,
    embedding_model,
):
    """
    Create Agent 2: Medicine Name Extraction Agent
    
    This agent extracts medicine names from queries and searches for them.
    Uses search_medicine_by_name tool from rag_tools.
    Returns a CompiledStateGraph (LangGraph) that can be invoked with messages.
    """
    from app.rag_tools import create_rag_tools
    
    # Create tools dict - pass None for rag_collection since we only need medicine search
    tools_dict = create_rag_tools(
        rag_collection=None,  # Not needed for medicine search
        medicines_collection=medicines_collection,
        embedding_model=embedding_model,
    )
    
    # Use only search_medicine_by_name tool
    tools = [tools_dict["search_medicine_by_name"]]
    
    system_prompt = """You are a medicine name extraction assistant. Your task is to:
1. Extract medicine names from the user's query
2. Search for these medicines in the database using search_medicine_by_name
3. Return a list of medicine IDs found (format: comma-separated list of 32-character hex IDs)

If multiple medicines are mentioned, find all of them.
Return ONLY the medicine IDs, one per line or comma-separated."""
    
    return create_agent(llm, tools=tools, system_prompt=system_prompt)


def create_full_instruction_agent(
    llm: BaseChatModel,
    rag_collection: chromadb.Collection,
    medicines_collection: chromadb.Collection,
    embedding_model,
    response_language: str = "English",
):
    """
    Create Agent 3: Full Instruction Agent
    
    This agent retrieves full instructions for medicines and generates comprehensive answers.
    Uses get_medicine_full_instruction and search_medical_information tools.
    Returns a CompiledStateGraph (LangGraph) that can be invoked with messages.
    """
    tools_dict = create_rag_tools(rag_collection, medicines_collection, embedding_model)
    # Use get_medicine_full_instruction and search_medical_information
    tools = [
        tools_dict["get_medicine_full_instruction"],
        tools_dict["search_medical_information"],
    ]
    
    system_prompt = f"""You are a medical information assistant. Your task is to provide comprehensive 
answers based on full medical instructions.

You have access to these tools:
- get_medicine_full_instruction: Get the full instruction text for a specific medicine by reading the MHT file directly
- search_medical_information: Search the medical instruction database using semantic search

Your workflow:
1. Use get_medicine_full_instruction to retrieve full instructions for the medicine IDs provided
2. Analyze the complete instruction text
3. If needed, use search_medical_information to find additional relevant information
4. Answer the user's question comprehensively based on the full instructions
5. Respond in {response_language}

Be thorough, accurate, and cite specific sections when possible."""
    
    return create_agent(llm, tools=tools, system_prompt=system_prompt)


def extract_medicine_ids(text: str) -> list[str]:
    """Extract medicine IDs (32-character hex strings) from text."""
    medicine_ids = re.findall(r'[A-F0-9]{32}', text.upper())
    # Remove duplicates while preserving order
    seen = set()
    return [mid for mid in medicine_ids if mid not in seen and not seen.add(mid)]
