"""Medical Instructions Indexing Script

Processes HTML and MHT files, extracts text content, chunks it, generates embeddings,
and stores in ChromaDB RAG collection with metadata linking back to medicines.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Import helper functions
from app.helpers import extract_text_from_html, parse_mht_file

# Import from ingestion module
from app.ingestion import (
    CHROMA_DIR,
    CHROMA_MEDICINES_COLLECTION,
    CHROMA_RAG_COLLECTION,
    HTML_DIR,
    MHT_DIR,
    load_metadata_from_chroma,
)

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
INDEXING_LIMIT = int(os.getenv("INDEXING_LIMIT", "0"))  # 0 = process all

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_medicine_id_from_filename(filename: str) -> Optional[str]:
    """Extract medicine ID from HTML/MHT filename"""
    # HTML files: {ID}.html
    # MHT files: {ID}.mht or UA{number}_{ID}.mht
    if filename.endswith(".html"):
        return filename[:-5]  # Remove .html
    elif filename.endswith(".mht"):
        # Try to extract ID from filename patterns
        # Pattern 1: {ID}.mht
        # Pattern 2: UA{number}_{ID}.mht
        name_without_ext = filename[:-4]
        # If it contains underscore, try to get the last part
        if "_" in name_without_ext:
            parts = name_without_ext.split("_")
            # Last part might be the ID (32 chars hex)
            if len(parts[-1]) == 32:
                return parts[-1]
        # Otherwise, if it's 32 chars, it's the ID
        if len(name_without_ext) == 32:
            return name_without_ext
    return None




def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks with overlap"""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary if not at end
        if end < text_length:
            # Look for sentence endings
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # Only break if we're not too early
                chunk = chunk[:break_point + 1]
                end = start + len(chunk)

        chunks.append(chunk.strip())
        start = end - chunk_overlap  # Overlap for context

    return chunks


def get_embedding_dimension(client: OpenAI, model: str) -> int:
    """Get embedding dimension for a model by making a test request"""
    try:
        response = client.embeddings.create(
            model=model,
            input="test",
        )
        dimension = len(response.data[0].embedding)
        logger.info(f"Detected embedding dimension for {model}: {dimension}")
        return dimension
    except Exception as e:
        logger.error(f"Error detecting embedding dimension: {e}")
        # Fallback to known dimensions
        if "3-large" in model:
            return 3072
        elif "3-small" in model or "ada-002" in model:
            return 1536
        else:
            logger.warning(f"Unknown model {model}, assuming 1536 dimensions")
            return 1536


def generate_embeddings(texts: list[str], client: OpenAI) -> tuple[list[list[float]], dict]:
    """Generate embeddings for a list of texts
    
    Returns:
        tuple: (embeddings list, usage info dict with 'total_tokens', 'prompt_tokens')
    """
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        
        # Extract usage information
        usage_info = {
            "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0,
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
        }
        
        return embeddings, usage_info
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [], {"total_tokens": 0, "prompt_tokens": 0}


def init_rag_collection(
    chroma_client: chromadb.Client,
    collection_name: str,
    openai_client: OpenAI,
    expected_dimension: int | None = None,
) -> chromadb.Collection:
    """Initialize or get RAG collection with dimension validation
    
    Args:
        chroma_client: ChromaDB client
        collection_name: Name of the collection
        openai_client: OpenAI client for dimension detection
        expected_dimension: Expected embedding dimension (if None, will detect from model)
    
    Returns:
        chromadb.Collection: The RAG collection
    """
    try:
        # Detect expected dimension if not provided
        if expected_dimension is None:
            expected_dimension = get_embedding_dimension(openai_client, OPENAI_EMBED_MODEL)
        
        # Check if collection exists
        try:
            existing_collection = chroma_client.get_collection(name=collection_name)
            
            # Check if collection has data
            existing_count = existing_collection.count()
            existing_dimension = None
            
            if existing_count > 0:
                # Get a sample embedding to check dimension
                sample_results = existing_collection.get(limit=1)
                if sample_results.get("embeddings") and len(sample_results["embeddings"]) > 0:
                    existing_dimension = len(sample_results["embeddings"][0])
            else:
                # For empty collections, try to get dimension from metadata or test with a dummy embedding
                # ChromaDB might have stored dimension in collection metadata
                # If we can't determine it, we'll delete and recreate to be safe
                try:
                    # Try to peek at collection metadata if available
                    collection_metadata = existing_collection.metadata if hasattr(existing_collection, 'metadata') else {}
                    if "embedding_dimension" in collection_metadata:
                        existing_dimension = int(collection_metadata["embedding_dimension"])
                        logger.info(f"Found dimension {existing_dimension} in collection metadata")
                except Exception:
                    # Can't determine dimension, will delete and recreate
                    pass
            
            # If dimension doesn't match (or we can't determine it for empty collection), delete and recreate
            if existing_dimension is not None and existing_dimension != expected_dimension:
                logger.warning("=" * 60)
                logger.warning("EMBEDDING DIMENSION MISMATCH!")
                logger.warning(f"Existing collection has dimension: {existing_dimension}")
                logger.warning(f"New model '{OPENAI_EMBED_MODEL}' has dimension: {expected_dimension}")
                logger.warning("=" * 60)
                logger.warning("Deleting existing collection and creating new one with correct dimension...")
                chroma_client.delete_collection(name=collection_name)
                logger.info("✅ Old collection deleted")
            elif existing_dimension is None and existing_count == 0:
                # Empty collection with unknown dimension - delete to be safe
                logger.warning(f"Empty collection '{collection_name}' found with unknown dimension.")
                logger.warning("Deleting and recreating with correct dimension...")
                chroma_client.delete_collection(name=collection_name)
                logger.info("✅ Old collection deleted")
            elif existing_dimension == expected_dimension:
                logger.info(f"RAG collection '{collection_name}' has {existing_count} chunks with dimension {existing_dimension} ✓")
                return existing_collection
            else:
                # Should not reach here, but return existing if dimension matches
                return existing_collection
                
        except Exception as e:
            # Collection doesn't exist, will create it
            logger.debug(f"Collection '{collection_name}' does not exist yet: {e}")
            pass
        
        # Create new collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "RAG document chunks with embeddings for medical instructions",
                "embedding_model": OPENAI_EMBED_MODEL,
                "embedding_dimension": str(expected_dimension),
            },
        )
        logger.info(f"RAG collection '{collection_name}' created with dimension {expected_dimension}")
        return collection
    except Exception as e:
        logger.error(f"Error initializing RAG collection: {e}")
        raise


def is_file_indexed(
    file_path: Path,
    medicine_id: str,
    rag_collection: chromadb.Collection,
) -> bool:
    """Check if a file has already been indexed"""
    try:
        # Normalize path for comparison (resolve to absolute path)
        normalized_path = str(file_path.resolve())
        
        # Query for chunks with this medicine_id and source_file
        # Try both normalized and original path formats
        results = rag_collection.get(
            where={"medicine_id": medicine_id},
            limit=100,  # Get more to check source_file matches
        )
        
        # Check if any chunk has matching source_file
        for metadata in (results.get("metadatas") or []):
            if metadata and metadata.get("source_file"):
                stored_path = metadata["source_file"]
                # Compare normalized paths
                if Path(stored_path).resolve() == Path(normalized_path).resolve():
                    return True
        
        return False
    except Exception as e:
        logger.warning(f"Error checking if file is indexed: {e}")
        return False


def index_file(
    file_path: Path,
    file_type: str,  # "html" or "mht"
    medicine_id: str,
    rag_collection: chromadb.Collection,
    openai_client: OpenAI,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """Index a single HTML or MHT file
    
    Args:
        skip_existing: If True, skip files that are already indexed
    
    Returns:
        tuple: (number of chunks indexed, tokens used)
    """
    # Check if already indexed
    if skip_existing and is_file_indexed(file_path, medicine_id, rag_collection):
        logger.info(f"⏭️  Skipping {file_type.upper()} file: {file_path.name} (already indexed)")
        return 0, 0
    
    logger.info(f"Processing {file_type.upper()} file: {file_path.name} (medicine: {medicine_id})")
    
    # Read and parse file
    if file_type == "mht":
        html_content = parse_mht_file(file_path)
        if not html_content:
            logger.warning(f"Could not extract HTML from MHT file: {file_path}")
            return 0, 0
        logger.debug(f"  Extracted HTML from MHT ({len(html_content):,} chars)")
    else:  # html
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="windows-1251") as f:
                    html_content = f.read()
            except Exception as e:
                logger.error(f"Error reading HTML file {file_path}: {e}")
                return 0, 0
        except Exception as e:
            logger.error(f"Error reading HTML file {file_path}: {e}")
            return 0, 0
        logger.debug(f"  Read HTML file ({len(html_content):,} chars)")

    # Extract text
    text = extract_text_from_html(html_content)
    if not text or len(text) < 50:  # Skip very short texts
        logger.warning(f"Text too short or empty from {file_path}")
        return 0, 0
    logger.debug(f"  Extracted text ({len(text):,} chars)")

    # Chunk text
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return 0, 0
    logger.info(f"  Created {len(chunks)} chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")

    # Generate embeddings in batches
    batch_size = 100  # OpenAI allows up to 2048 inputs per request
    all_embeddings = []
    all_chunk_ids = []
    all_metadatas = []
    total_tokens_used = 0

    # Normalize file path for consistent storage
    normalized_file_path = str(file_path.resolve())
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{medicine_id}_{file_type}_{i}"
        all_chunk_ids.append(chunk_id)
        all_metadatas.append({
            "medicine_id": medicine_id,
            "source_file": normalized_file_path,
            "file_type": file_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

    # Generate embeddings in batches
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info(f"  Generating embeddings for {len(chunks)} chunks in {num_batches} batch(es)...")
    
    for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size), 1):
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        batch_embeddings, usage_info = generate_embeddings(batch_chunks, openai_client)
        all_embeddings.extend(batch_embeddings)
        total_tokens_used += usage_info["total_tokens"]
        
        if num_batches > 1:
            logger.debug(f"    Batch {batch_idx}/{num_batches}: {len(batch_chunks)} chunks, {usage_info['total_tokens']} tokens")

    # Store in ChromaDB
    if all_embeddings:
        rag_collection.add(
            ids=all_chunk_ids,
            embeddings=all_embeddings,
            documents=chunks,
            metadatas=all_metadatas,
        )
        logger.info(
            f"✅ Indexed {len(chunks)} chunks from {file_path.name} "
            f"(medicine: {medicine_id}, tokens: {total_tokens_used:,})"
        )
        return len(chunks), total_tokens_used

    return 0, 0


def reset_indexing_progress(chroma_client: chromadb.Client, collection_name: str) -> None:
    """Reset indexing progress: delete the entire RAG collection"""
    logger.warning("=" * 60)
    logger.warning("RESETTING INDEXING PROGRESS")
    logger.warning("=" * 60)
    
    try:
        # Try to get collection to check if it exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
            count = collection.count()
            logger.info(f"Found collection '{collection_name}' with {count} chunks")
            
            # Delete the entire collection
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"✅ Collection '{collection_name}' deleted completely")
        except Exception as e:
            # Collection doesn't exist or already deleted
            logger.info(f"Collection '{collection_name}' does not exist or already deleted: {e}")
    except Exception as e:
        logger.error(f"Error resetting indexing progress: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("Indexing progress reset complete!")
    logger.info("=" * 60)


def main() -> None:
    """Main indexing function"""
    parser = argparse.ArgumentParser(description="Medical instructions indexing script")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset indexing progress: delete all chunks from RAG collection",
    )
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in .env file.")
        return

    logger.info("Starting medical instructions indexing")

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Handle reset (must be done before initializing collection)
    if args.reset:
        reset_indexing_progress(chroma_client, CHROMA_RAG_COLLECTION)
        logger.info("Exiting after reset. Run without --reset to start indexing.")
        return
    
    rag_collection = init_rag_collection(chroma_client, CHROMA_RAG_COLLECTION, openai_client)

    # Load medicines metadata to get file paths
    medicines_collection = chroma_client.get_collection(CHROMA_MEDICINES_COLLECTION)
    medicines_metadata = load_metadata_from_chroma(medicines_collection)

    logger.info(f"Found {len(medicines_metadata)} medicines in database")

    # Collect files to index
    files_to_index: list[tuple[Path, str, str]] = []  # (file_path, file_type, medicine_id)

    for medicine_id, metadata in medicines_metadata.items():
        # Add HTML file if exists
        if metadata.file_path:
            html_path = Path(metadata.file_path)
            if html_path.exists():
                files_to_index.append((html_path, "html", medicine_id))

        # Add MHT file if exists
        if metadata.mht_file_path:
            mht_path = Path(metadata.mht_file_path)
            if mht_path.exists():
                files_to_index.append((mht_path, "mht", medicine_id))

    # Apply limit if set
    if INDEXING_LIMIT > 0:
        original_count = len(files_to_index)
        files_to_index = files_to_index[:INDEXING_LIMIT]
        logger.info(f"Limited indexing to {len(files_to_index)} files (out of {original_count} total)")

    logger.info(f"Found {len(files_to_index)} files to index")
    logger.info(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
    logger.info(f"Embedding model: {OPENAI_EMBED_MODEL}")
    logger.info("=" * 60)

    # Index files
    total_chunks = 0
    indexed_files = 0
    total_tokens = 0
    html_files_count = 0
    mht_files_count = 0

    with tqdm(total=len(files_to_index), desc="Indexing files", unit="file") as pbar:
        for file_path, file_type, medicine_id in files_to_index:
            chunks_count, tokens_used = index_file(
                file_path, file_type, medicine_id, rag_collection, openai_client, skip_existing=True
            )
            if chunks_count > 0:
                indexed_files += 1
                total_chunks += chunks_count
                total_tokens += tokens_used
                if file_type == "html":
                    html_files_count += 1
                else:
                    mht_files_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "chunks": total_chunks,
                "tokens": f"{total_tokens:,}",
                "files": indexed_files
            })

    # Summary
    logger.info("=" * 60)
    logger.info("Indexing complete!")
    logger.info(f"Files processed: {indexed_files} / {len(files_to_index)}")
    logger.info(f"  - HTML files: {html_files_count}")
    logger.info(f"  - MHT files: {mht_files_count}")
    logger.info(f"Total chunks created: {total_chunks:,}")
    logger.info(f"Total tokens used: {total_tokens:,}")
    if total_chunks > 0:
        avg_tokens_per_chunk = total_tokens / total_chunks
        logger.info(f"Average tokens per chunk: {avg_tokens_per_chunk:.1f}")
    logger.info(f"RAG collection: {CHROMA_RAG_COLLECTION}")
    logger.info(f"ChromaDB location: {CHROMA_DIR}")
    
    # Estimate cost (text-embedding-3-small: $0.02 per 1M tokens, text-embedding-3-large: $0.13 per 1M tokens)
    if total_tokens > 0:
        cost_per_million = 0.13 if "3-large" in OPENAI_EMBED_MODEL else 0.02
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        logger.info(f"Estimated cost: ${estimated_cost:.4f} (at ${cost_per_million:.2f} per 1M tokens for {OPENAI_EMBED_MODEL})")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

