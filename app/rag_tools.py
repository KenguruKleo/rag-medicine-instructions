"""
LangChain tools for RAG search and medicine information retrieval.
"""

from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.tools import StructuredTool

from app.helpers import extract_text_from_html, parse_mht_file


def create_rag_tools(
    rag_collection: chromadb.Collection | None,
    medicines_collection: chromadb.Collection,
    embedding_model,
) -> dict:
    """
    Create LangChain tools for RAG operations.
    
    Args:
        rag_collection: ChromaDB collection with instruction chunks (can be None if only medicine search is needed)
        medicines_collection: ChromaDB collection with medicine metadata
        embedding_model: Embedding model instance
    
    Returns:
        Dictionary of LangChain tools with keys:
        - "search_medical_information": Semantic search tool (requires rag_collection)
        - "search_medicine_by_name": Medicine name search tool
        - "get_medicine_full_instruction": Full instruction retrieval tool
    """
    
    def search_medical_information(query: str, n_results: int = 3) -> str:
        """Search medical information using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return (default: 3)
        
        Returns:
            JSON string with search results
        """
        import json
        try:
            # Generate embedding for query
            query_embedding = embedding_model.embed_query(query)
            
            # Search in ChromaDB
            results = rag_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    document = results["documents"][0][i]
                    # Truncate to 1000 chars to prevent token overflow
                    truncated_doc = document[:1000] + "..." if len(document) > 1000 else document
                    formatted_results.append({
                        "chunk_id": results["ids"][0][i],
                        "document": truncated_doc,
                        "medicine_id": results["metadatas"][0][i].get("medicine_id", "N/A"),
                        "source_file": results["metadatas"][0][i].get("source_file", "N/A"),
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                    })
            
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def search_medicine_by_name(medicine_name: str, n_results: int = 3) -> str:
        """Search for medicines by name using semantic search.
        
        Args:
            medicine_name: Name of the medicine to search for
            n_results: Number of results to return (default: 3)
        
        Returns:
            JSON string with medicine search results
        """
        import json
        try:
            # Generate embedding for query
            query_embedding = embedding_model.embed_query(medicine_name)
            
            # Search in ChromaDB
            results = medicines_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]
                    formatted_results.append({
                        "medicine_id": results["ids"][0][i],
                        "ukrainian_name": metadata.get("ukrainian_name", "N/A"),
                        "international_name": metadata.get("international_name", "N/A"),
                        "medicinal_product_name": metadata.get("medicinal_product_name", "N/A"),
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                    })
            
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    

    def get_medicine_full_instruction(medicine_id: str) -> str:
        """Get full instruction text for a specific medicine by reading the MHT file directly.
        
        Args:
            medicine_id: ID of the medicine (32-character hex string)
            max_chunks: Not used (kept for compatibility), full instruction is returned
        
        Returns:
            JSON string with full instruction text from MHT file
        """
        import json
        if rag_collection is None:
            return json.dumps({"error": "RAG collection is not available"})
        try:
            # Get medicine metadata to find MHT file path
            medicine_result = medicines_collection.get(ids=[medicine_id])
            if not medicine_result["ids"] or not medicine_result["metadatas"]:
                return json.dumps({"error": f"Medicine {medicine_id} not found"})
            
            medicine_metadata = medicine_result["metadatas"][0]
            mht_file_path = medicine_metadata.get("mht_file_path", "")
            html_file_path = medicine_metadata.get("html_file_path", "")
            
            # Try to find MHT file
            mht_path = None
            
            # First, try the path from metadata
            if mht_file_path:
                mht_path = Path(mht_file_path)
                if not mht_path.exists():
                    mht_path = None
            
            # If not found, try to find MHT file in data/mht directory
            if not mht_path:
                data_dir = Path("data/mht")
                if data_dir.exists():
                    # Try exact match first
                    exact_path = data_dir / f"{medicine_id}.mht"
                    if exact_path.exists():
                        mht_path = exact_path
                    else:
                        # Try pattern matching (UA*_{medicine_id}.mht)
                        matches = list(data_dir.glob(f"*_{medicine_id}.mht"))
                        if matches:
                            mht_path = matches[0]
            
            # If still not found, try HTML file as fallback
            if not mht_path and html_file_path:
                html_path = Path(html_file_path)
                if html_path.exists():
                    # Use HTML file instead
                    html_content = html_path.read_text(encoding="utf-8", errors="ignore")
                    full_text = extract_text_from_html(html_content)
                    
                    if len(full_text) > 50000:
                        full_text = full_text[:50000] + "\n\n... (текст обрізано для економії токенів)"
                    
                    source_file = str(html_path)
                    if Path(source_file).is_absolute():
                        try:
                            data_dir = Path("data").resolve()
                            if str(source_file).startswith(str(data_dir)):
                                source_file = str(Path(source_file).relative_to(data_dir.parent))
                        except (ValueError, AttributeError):
                            pass
                    
                    formatted_result = {
                        "medicine_id": medicine_id,
                        "source_file": source_file,
                        "full_instruction": full_text,
                        "instruction_length": len(full_text),
                        "file_type": "html",
                    }
                    
                    return json.dumps(formatted_result, ensure_ascii=False, indent=2)
            
            if not mht_path:
                return json.dumps({"error": f"MHT or HTML file not found for medicine {medicine_id}"})
            
            # Parse MHT file
            html_content = parse_mht_file(mht_path)
            if not html_content:
                return json.dumps({"error": f"Failed to parse MHT file {mht_path}"})
            
            # Extract text from HTML
            full_text = extract_text_from_html(html_content)
            
            # Truncate to reasonable size to prevent token overflow (keep first 50000 chars)
            # This is still much more than chunks, but prevents issues
            if len(full_text) > 50000:
                full_text = full_text[:50000] + "\n\n... (текст обрізано для економії токенів)"
            
            # Convert absolute path to relative if needed
            source_file = str(mht_path)
            if Path(source_file).is_absolute():
                try:
                    data_dir = Path("data").resolve()
                    if str(source_file).startswith(str(data_dir)):
                        source_file = str(Path(source_file).relative_to(data_dir.parent))
                except (ValueError, AttributeError):
                    pass
            
            formatted_result = {
                "medicine_id": medicine_id,
                "source_file": source_file,
                "full_instruction": full_text,
                "instruction_length": len(full_text),
                "file_type": "mht",
            }
            
            return json.dumps(formatted_result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    # Create tool instances
    search_tool = StructuredTool.from_function(
        func=search_medical_information,
        name="search_medical_information",
        description="Search medical information using semantic search. Use this to find relevant information from the medical instruction database. Input: query string and optional n_results (default 3).",
    )
    
    medicine_search_tool = StructuredTool.from_function(
        func=search_medicine_by_name,
        name="search_medicine_by_name",
        description="Search for medicines by name. Use this to find medicine IDs when you know the medicine name. Input: medicine name string and optional n_results (default 3).",
    )
    
    get_chunks_tool = StructuredTool.from_function(
        func=get_medicine_full_instruction,
        name="get_medicine_full_instruction",
        description="Get the full instruction text for a specific medicine by reading the MHT file directly. This returns the complete instruction text from the source file, not chunks. Use this to retrieve comprehensive information about a medicine. Input: medicine_id (32-character hex string). The max_chunks parameter is kept for compatibility but ignored - full instruction is always returned.",
    )
    
    # Return as dictionary for easy access by name
    # Only include tools that are available (based on rag_collection)
    tools_dict = {
        "search_medicine_by_name": medicine_search_tool,
    }
    
    if rag_collection is not None:
        tools_dict["search_medical_information"] = search_tool
        tools_dict["get_medicine_full_instruction"] = get_chunks_tool
    
    return tools_dict

