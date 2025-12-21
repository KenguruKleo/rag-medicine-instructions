"""
Streamlit web application for RAG-based medical instructions search.

This application provides a chat interface for querying Ukrainian medical instructions
using semantic search and OpenAI LLM for generating responses.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "storage/chroma"))
CHROMA_RAG_COLLECTION = os.getenv("CHROMA_RAG_COLLECTION", "instruction_chunks")
CHROMA_MEDICINES_COLLECTION = os.getenv("CHROMA_MEDICINES_COLLECTION", "medicines")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")

# Page configuration
st.set_page_config(
    page_title="Medical Instructions RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize clients (cached)
@st.cache_resource
def init_clients():
    """Initialize ChromaDB and OpenAI clients."""
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not set. Please configure it in .env file.")
        st.stop()
    
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Check if collections exist
        try:
            rag_collection = chroma_client.get_collection(CHROMA_RAG_COLLECTION)
            medicines_collection = chroma_client.get_collection(CHROMA_MEDICINES_COLLECTION)
        except Exception as e:
            st.error(f"❌ ChromaDB collections not found: {e}")
            st.info("Please run ingestion and indexing first.")
            st.stop()
        
        return {
            "chroma_client": chroma_client,
            "openai_client": openai_client,
            "rag_collection": rag_collection,
            "medicines_collection": medicines_collection,
        }
    except Exception as e:
        st.error(f"❌ Error initializing clients: {e}")
        st.stop()


def search_instructions(
    query: str,
    rag_collection: chromadb.Collection,
    openai_client: OpenAI,
    n_results: int = 5,
) -> list[dict]:
    """Search medical instructions using semantic search."""
    try:
        # Generate embedding for query
        response = openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=query,
        )
        query_embedding = response.data[0].embedding

        # Search in ChromaDB
        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
            )

        return formatted_results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []


def get_medicine_info(
    medicine_id: str,
    medicines_collection: chromadb.Collection,
) -> dict:
    """Get medicine information from medicines collection."""
    try:
        result = medicines_collection.get(ids=[medicine_id])
        if result["ids"] and result["metadatas"]:
            return result["metadatas"][0]
        return {}
    except Exception:
        return {}


def create_file_link(file_path: str, source_number: int = None) -> str:
    """Create a link URL for a file path with optional source number."""
    if file_path == "N/A":
        return "N/A" if source_number is None else f"Source {source_number}: N/A"
    
    # Convert absolute path to relative path from project root
    rel_path = file_path
    if Path(file_path).is_absolute():
        # Try to make it relative to current working directory
        try:
            rel_path = str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            # If can't make relative, try relative to data directory
            try:
                data_dir_abs = (Path.cwd() / "data").resolve()
                rel_path = str(Path(file_path).relative_to(data_dir_abs))
                rel_path = f"data/{rel_path}"
            except ValueError:
                # If still can't make relative, use absolute path
                rel_path = file_path
    
    # Create link URL (encode path for URL safety)
    link_url = f"/view_file?file={quote(rel_path)}"
    file_name = Path(file_path).name
    
    # Format link text with source number if provided
    if source_number is not None:
        link_text = f"Source {source_number}: {file_name}"
    else:
        link_text = file_name
    
    return f"[{link_text}]({link_url})"


def replace_source_mentions(text: str, sources: list[dict]) -> str:
    """Replace 'Source N' mentions in text with clickable file links.
    Groups sources by file to avoid duplicates."""
    import re
    
    # Group sources by file path
    file_to_sources = {}
    for i, source in enumerate(sources, 1):
        file_path = source.get("source_file", "N/A")
        if file_path not in file_to_sources:
            file_to_sources[file_path] = []
        file_to_sources[file_path].append(i)
    
    # Create a mapping of source numbers to file links
    # If multiple sources point to same file, use the first source number
    source_links = {}
    for file_path, source_numbers in file_to_sources.items():
        # Use the first source number for this file
        first_source_num = source_numbers[0]
        # If only one source points to this file, don't include source number in link
        if len(source_numbers) == 1:
            source_links[first_source_num] = create_file_link(file_path, source_number=None)
        else:
            # Multiple sources point to same file - use first source number
            source_links[first_source_num] = create_file_link(file_path, source_number=first_source_num)
        
        # Map all source numbers pointing to this file to the same link
        for source_num in source_numbers:
            source_links[source_num] = source_links[first_source_num]
    
    # Pattern to match "Source N" (case-insensitive, with optional punctuation)
    # Matches: "Source 1", "source 2", "Source 1,", "Source 1:", etc.
    pattern = r'\bSource\s+(\d+)\b'
    
    # Track which sources we've already replaced to avoid duplicates
    replaced_sources = set()
    
    def replace_match(match):
        source_num = int(match.group(1))
        if source_num in source_links:
            link = source_links[source_num]
            # Check if we've already replaced a source pointing to the same file
            file_path = sources[source_num - 1].get("source_file", "N/A")
            if file_path in file_to_sources:
                # Get all source numbers for this file
                file_sources = file_to_sources[file_path]
                # If this is not the first source for this file, and we've already replaced the first one
                if source_num != file_sources[0] and file_sources[0] in replaced_sources:
                    # Return empty string to remove duplicate
                    return ""
            replaced_sources.add(source_num)
            return link
        return match.group(0)  # Return original if source number not found
    
    # Replace all matches
    result = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    
    # Clean up any double commas or spaces left after removing duplicates
    result = re.sub(r',\s*,', ',', result)  # Remove double commas
    result = re.sub(r',\s*\.', '.', result)  # Remove comma before period
    result = re.sub(r'\s+', ' ', result)  # Remove extra spaces
    result = re.sub(r',\s*$', '', result)  # Remove trailing comma
    
    return result


def ask_rag_question(
    query: str,
    rag_collection: chromadb.Collection,
    medicines_collection: chromadb.Collection,
    openai_client: OpenAI,
    response_language: str = "English",
    n_results: int = 3,
    max_context_chars: int = 2000,
) -> dict:
    """
    Ask a question in any language, get response in specified language.
    
    Args:
        query: Question in any language (e.g., Ukrainian)
        response_language: Language for the response (e.g., "English", "Ukrainian")
        n_results: Number of relevant chunks to retrieve
        max_context_chars: Maximum characters of context to include
    
    Returns:
        dict with 'answer', 'sources', 'chunks_used', 'error'
    """
    # Step 1: Semantic search to find relevant chunks
    search_results = search_instructions(query, rag_collection, openai_client, n_results=n_results)

    if not search_results:
        return {"error": "No relevant information found"}

    # Step 2: Build context from search results
    context_parts = []
    sources = []

    for i, result in enumerate(search_results, 1):
        chunk_text = result["document"]
        medicine_id = result["metadata"].get("medicine_id", "N/A")
        source_file = result["metadata"].get("source_file", "N/A")

        # Truncate if too long for context
        truncated_chunk = chunk_text
        if len(chunk_text) > max_context_chars // n_results:
            truncated_chunk = chunk_text[: max_context_chars // n_results] + "..."

        context_parts.append(f"[Source {i} - Medicine ID: {medicine_id}]\n{truncated_chunk}")
        sources.append(
            {
                "medicine_id": medicine_id,
                "source_file": source_file,
                "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                "file_type": result["metadata"].get("file_type", "N/A"),
                "chunk_text": chunk_text,  # Full chunk text for display
            }
        )

    context = "\n\n".join(context_parts)

    # Step 3: Build prompt for LLM
    system_prompt = f"""You are a medical information assistant. You help users understand medical instructions.
The medical instructions are in Ukrainian, but you should respond in {response_language}.

Your task:
1. Understand the Ukrainian medical instruction content provided
2. Answer the user's question based on the provided context
3. Respond clearly and accurately in {response_language}
4. If the context doesn't contain enough information, say so
5. Always cite which source(s) you used (Source 1, Source 2, etc.)

Be professional, accurate, and helpful."""

    user_prompt = f"""Question: {query}

Relevant medical instruction context (in Ukrainian):
{context}

Please answer the question in {response_language} based on the provided context."""

    # Step 4: Get response from LLM
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Note: gpt-5-nano only supports default temperature (1), custom values are not supported
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(search_results),
            "model": OPENAI_LLM_MODEL,
            "tokens_used": response.usage.total_tokens if hasattr(response, "usage") else None,
        }
    except Exception as e:
        return {"error": f"Error generating response: {e}"}


def main():
    """Main Streamlit application."""
    # Initialize clients
    clients = init_clients()
    chroma_client = clients["chroma_client"]
    openai_client = clients["openai_client"]
    rag_collection = clients["rag_collection"]
    medicines_collection = clients["medicines_collection"]

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check collection status
        try:
            total_chunks = rag_collection.count()
            st.success(f"✅ {total_chunks:,} chunks indexed")
        except Exception:
            st.error("❌ RAG collection not available")
        
        # Response language selection
        response_language = st.selectbox(
            "Response Language",
            ["English", "Ukrainian"],
            index=0,
            help="Language for the AI response",
        )
        
        # Number of results
        n_results = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant chunks to retrieve",
        )
        
        st.divider()
        
        # Model information
        st.caption(f"LLM: {OPENAI_LLM_MODEL}")
        st.caption(f"Embedding: {OPENAI_EMBED_MODEL}")

    # Main content
    st.title("💊 Medical Instructions RAG")
    st.markdown(
        """
        Ask questions about Ukrainian medical instructions in any language.
        The system will search through indexed instructions and provide answers.
        """
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Replace Source mentions with links for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                content_with_links = replace_source_mentions(
                    message["content"],
                    message.get("sources", [])
                )
                st.markdown(content_with_links)
            else:
                st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        medicine_info = get_medicine_info(
                            source["medicine_id"], medicines_collection
                        )
                        st.markdown(f"**Source {i}:**")
                        if medicine_info:
                            st.markdown(
                                f"- **Medicine:** {medicine_info.get('ukrainian_name', 'N/A')}"
                            )
                            st.markdown(
                                f"- **International name:** {medicine_info.get('international_name', 'N/A')}"
                            )
                        # Show file name as link
                        file_path = source['source_file']
                        file_name = Path(file_path).name if file_path != "N/A" else "N/A"
                        
                        if file_path != "N/A":
                            # Create link to view file page
                            # Convert absolute path to relative path from project root
                            rel_path = file_path
                            if Path(file_path).is_absolute():
                                # Try to make it relative to data directory
                                data_dir = Path("data")
                                try:
                                    rel_path = str(Path(file_path).relative_to(Path.cwd() / data_dir))
                                    rel_path = f"data/{rel_path}"
                                except ValueError:
                                    # If can't make relative, use as is
                                    rel_path = file_path
                            
                            # Create link URL
                            link_url = f"/view_file?file={rel_path}"
                            st.markdown(f"- **File:** [{file_name}]({link_url})")
                        else:
                            st.markdown(f"- **File:** {file_name}")
                        
                        st.markdown(f"- **Chunk:** {source['chunk_index']}")
                        # Show chunk text instead of file type
                        if source.get("chunk_text"):
                            st.markdown("**Chunk text:**")
                            st.text_area(
                                f"Chunk {i} content",
                                value=source["chunk_text"],
                                height=150,
                                key=f"history_msg_{msg_idx}_chunk_{i}",
                                label_visibility="collapsed",
                            )
                        if i < len(message["sources"]):
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about medical instructions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                result = ask_rag_question(
                    query=prompt,
                    rag_collection=rag_collection,
                    medicines_collection=medicines_collection,
                    openai_client=openai_client,
                    response_language=response_language,
                    n_results=n_results,
                )

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Replace Source mentions with clickable file links
                    answer_with_links = replace_source_mentions(
                        result["answer"], 
                        result.get("sources", [])
                    )
                    # Display answer
                    st.markdown(answer_with_links)
                    
                    # Show metadata
                    with st.expander("ℹ️ Response Details"):
                        st.markdown(f"**Chunks used:** {result['chunks_used']}")
                        if result.get("tokens_used"):
                            st.markdown(f"**Tokens used:** {result['tokens_used']:,}")
                        st.markdown(f"**Model:** {result['model']}")
                    
                    # Show sources immediately after response
                    if result.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(result["sources"], 1):
                                medicine_info = get_medicine_info(
                                    source["medicine_id"], medicines_collection
                                )
                                st.markdown(f"**Source {i}:**")
                                if medicine_info:
                                    st.markdown(
                                        f"- **Medicine:** {medicine_info.get('ukrainian_name', 'N/A')}"
                                    )
                                    st.markdown(
                                        f"- **International name:** {medicine_info.get('international_name', 'N/A')}"
                                    )
                                # Show file name as link
                                file_path = source['source_file']
                                file_name = Path(file_path).name if file_path != "N/A" else "N/A"
                                
                                if file_path != "N/A":
                                    # Create link to view file page
                                    # Convert absolute path to relative path from project root
                                    rel_path = file_path
                                    if Path(file_path).is_absolute():
                                        # Try to make it relative to current working directory
                                        try:
                                            rel_path = str(Path(file_path).relative_to(Path.cwd()))
                                        except ValueError:
                                            # If can't make relative, try relative to data directory
                                            try:
                                                data_dir_abs = (Path.cwd() / "data").resolve()
                                                rel_path = str(Path(file_path).relative_to(data_dir_abs))
                                                rel_path = f"data/{rel_path}"
                                            except ValueError:
                                                # If still can't make relative, use absolute path
                                                rel_path = file_path
                                    
                                    # Create link URL (encode path for URL safety)
                                    link_url = f"/view_file?file={quote(rel_path)}"
                                    st.markdown(f"- **File:** [{file_name}]({link_url})")
                                else:
                                    st.markdown(f"- **File:** {file_name}")
                                
                                st.markdown(f"- **Chunk:** {source['chunk_index']}")
                                # Show chunk text instead of file type
                                if source.get("chunk_text"):
                                    st.markdown("**Chunk text:**")
                                    # Use message count for unique key
                                    msg_count = len(st.session_state.messages)
                                    st.text_area(
                                        f"Chunk {i} content",
                                        value=source["chunk_text"],
                                        height=150,
                                        key=f"new_msg_{msg_count}_chunk_{i}",
                                        label_visibility="collapsed",
                                    )
                                if i < len(result["sources"]):
                                    st.divider()

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer", result.get("error", "No response")),
                "sources": result.get("sources", []),
            }
        )


if __name__ == "__main__":
    main()

