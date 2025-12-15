"""Medical Instructions Indexing Script

Processes HTML and MHT files, extracts text content, chunks it, generates embeddings,
and stores in ChromaDB RAG collection with metadata linking back to medicines.
"""

from __future__ import annotations

import email
import logging
import os
import quopri
import re
from email import policy
from pathlib import Path
from typing import Optional

import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

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


def parse_mht_file(mht_path: Path) -> Optional[str]:
    """Parse MHT file and extract HTML content"""
    try:
        with open(mht_path, "rb") as f:
            msg = email.message_from_bytes(f.read(), policy=policy.default)

        # Find the main HTML part
        html_content = None
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                # Get encoding
                encoding = part.get_content_charset() or "utf-8"
                transfer_encoding = part.get("Content-Transfer-Encoding", "").lower()

                # Get payload
                if transfer_encoding == "quoted-printable":
                    # Handle quoted-printable encoding
                    payload = part.get_payload()
                    if isinstance(payload, str):
                        # Decode quoted-printable
                        payload = quopri.decodestring(payload.encode()).decode(encoding, errors="ignore")
                        html_content = payload
                    else:
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                html_content = payload.decode(encoding)
                            except UnicodeDecodeError:
                                html_content = payload.decode("utf-8", errors="ignore")
                else:
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            html_content = payload.decode(encoding)
                        except UnicodeDecodeError:
                            try:
                                html_content = payload.decode("windows-1251")
                            except UnicodeDecodeError:
                                html_content = payload.decode("utf-8", errors="ignore")
                break

        return html_content
    except Exception as e:
        logger.error(f"Error parsing MHT file {mht_path}: {e}")
        return None


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content"""
    try:
        soup = BeautifulSoup(html_content, "lxml")
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return ""


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


def init_rag_collection(chroma_client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """Initialize or get RAG collection"""
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document chunks with embeddings for medical instructions"},
        )
        logger.info(f"RAG collection '{collection_name}' ready")
        return collection
    except Exception as e:
        logger.error(f"Error initializing RAG collection: {e}")
        raise


def index_file(
    file_path: Path,
    file_type: str,  # "html" or "mht"
    medicine_id: str,
    rag_collection: chromadb.Collection,
    openai_client: OpenAI,
) -> tuple[int, int]:
    """Index a single HTML or MHT file
    
    Returns:
        tuple: (number of chunks indexed, tokens used)
    """
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

    for i, chunk in enumerate(chunks):
        chunk_id = f"{medicine_id}_{file_type}_{i}"
        all_chunk_ids.append(chunk_id)
        all_metadatas.append({
            "medicine_id": medicine_id,
            "source_file": str(file_path),
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


def main() -> None:
    """Main indexing function"""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Please set it in .env file.")
        return

    logger.info("Starting medical instructions indexing")

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    rag_collection = init_rag_collection(chroma_client, CHROMA_RAG_COLLECTION)

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
            chunks_count, tokens_used = index_file(file_path, file_type, medicine_id, rag_collection, openai_client)
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
    
    # Estimate cost (text-embedding-3-small: $0.02 per 1M tokens)
    if total_tokens > 0:
        estimated_cost = (total_tokens / 1_000_000) * 0.02
        logger.info(f"Estimated cost: ${estimated_cost:.4f} (at $0.02 per 1M tokens)")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

