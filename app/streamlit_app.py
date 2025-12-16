"""
Streamlit web application for RAG-based medical instructions search.

This application provides a chat interface for querying Ukrainian medical instructions
using semantic search and OpenAI LLM for generating responses.
"""

import os
import sys
from pathlib import Path
from typing import Optional

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
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

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
        response_language: Language for the response (e.g., "English", "Ukrainian", "Russian")
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

        # Truncate if too long
        if len(chunk_text) > max_context_chars // n_results:
            chunk_text = chunk_text[: max_context_chars // n_results] + "..."

        context_parts.append(f"[Source {i} - Medicine ID: {medicine_id}]\n{chunk_text}")
        sources.append(
            {
                "medicine_id": medicine_id,
                "source_file": source_file,
                "chunk_index": result["metadata"].get("chunk_index", "N/A"),
                "file_type": result["metadata"].get("file_type", "N/A"),
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
            temperature=0.3,  # Lower temperature for more factual responses
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
            ["English", "Ukrainian", "Russian"],
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
                        st.markdown(f"- **File:** `{source['source_file']}`")
                        st.markdown(f"- **Type:** {source['file_type']}")
                        st.markdown(f"- **Chunk:** {source['chunk_index']}")
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
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Show metadata
                    with st.expander("ℹ️ Response Details"):
                        st.markdown(f"**Chunks used:** {result['chunks_used']}")
                        if result.get("tokens_used"):
                            st.markdown(f"**Tokens used:** {result['tokens_used']:,}")
                        st.markdown(f"**Model:** {result['model']}")

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

