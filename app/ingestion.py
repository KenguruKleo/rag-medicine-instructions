"""Medical Instructions Ingestion Script

Reads CSV registry, extracts medicine IDs and names, fetches HTML instruction pages,
and stores them locally with metadata.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet
import chromadb
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
CSV_PATH = Path(os.getenv("CSV_PATH", "reestr.csv"))
HTML_DIR = Path(os.getenv("HTML_DIR", "data/html"))
MHT_DIR = Path(os.getenv("MHT_DIR", "data/mht"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "storage/chroma"))
CHROMA_MEDICINES_COLLECTION = os.getenv("CHROMA_MEDICINES_COLLECTION", "medicines")
CHROMA_RAG_COLLECTION = os.getenv("CHROMA_RAG_COLLECTION", "instruction_chunks")
BASE_URL = "http://www.drlz.com.ua/ibp/ddsite.nsf/all/shlz1?opendocument&stype={}"
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
# Limit for testing purposes (set to None or 0 to process all items)
LIMIT = int(os.getenv("INGESTION_LIMIT", "100"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MedicineMetadata:
    """Metadata for a medicine entry"""

    id: str
    ukrainian_name: str
    international_name: str
    medicinal_product_name: str
    file_path: str  # HTML file path
    mht_file_path: str  # MHT file path
    fetch_timestamp: str
    status: str  # "success" or "failed"
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet"""
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
        confidence = result["confidence"]
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        return encoding or "windows-1251"


def read_csv_with_encoding(file_path: Path) -> list[dict]:
    """Read CSV file, handling Windows-1251 encoding"""
    encoding = detect_encoding(file_path)

    # Try to read with detected encoding, fallback to windows-1251
    for enc in [encoding, "windows-1251", "utf-8"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f, delimiter=";")
                rows = list(reader)
                logger.info(f"Successfully read {len(rows)} rows with encoding: {enc}")
                return rows
        except (UnicodeDecodeError, UnicodeError) as e:
            logger.warning(f"Failed to read with encoding {enc}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error reading CSV with encoding {enc}: {e}")
            raise

    raise ValueError(f"Could not read CSV file {file_path} with any encoding")


def extract_medicine_data(row: dict) -> Optional[dict]:
    """Extract medicine ID and names from CSV row"""
    # Column names based on the CSV structure:
    # Column 1: ID
    # Column 2: "Торгівельне найменування" (Trade name / Ukrainian name)
    # Column 3: "Міжнародне непатентоване найменування" (International name)
    # Column 4: "Форма випуску" (Release form) - not used for names

    # Get ID
    id_value = row.get("ID", "").strip().strip('"')
    if not id_value:
        return None

    # Extract names by column name (UTF-8)
    ukrainian_name = row.get("Торгівельне найменування", "").strip().strip('"')
    international_name = row.get("Міжнародне непатентоване найменування", "").strip().strip('"')
    medicinal_product_name = ukrainian_name  # Trade name is the medicinal product name
    instruction_url = row.get("URL інструкції", "").strip().strip('"')

    # Fallback: if column names don't match (encoding issue), try by position
    if not ukrainian_name:
        values = list(row.values())
        if len(values) >= 3:
            ukrainian_name = str(values[1]).strip().strip('"')
            international_name = str(values[2]).strip().strip('"')
            medicinal_product_name = ukrainian_name
        # URL is typically in column 42 (index 41)
        if len(values) > 41:
            instruction_url = str(values[41]).strip().strip('"')

    return {
        "id": id_value,
        "ukrainian_name": ukrainian_name,
        "international_name": international_name,
        "medicinal_product_name": medicinal_product_name,
        "instruction_url": instruction_url,
    }


def fetch_html(id_value: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Fetch HTML content for a medicine ID"""
    url = BASE_URL.format(id_value)

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            response.raise_for_status()

            # Check if response is actually HTML
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type or "text" in content_type:
                return response.text
            else:
                logger.warning(f"Unexpected content type for {id_value}: {content_type}")
                return response.text  # Return anyway, might still be HTML

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed for {id_value}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to fetch {id_value} after {retries} attempts: {e}")
                return None

    return None


def save_html_file(id_value: str, html_content: str, html_dir: Path) -> Path:
    """Save HTML content to file"""
    html_dir.mkdir(parents=True, exist_ok=True)
    file_path = html_dir / f"{id_value}.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return file_path


def download_mht_file(url: str, id_value: str, mht_dir: Path, retries: int = MAX_RETRIES) -> Optional[Path]:
    """Download MHT file from URL"""
    if not url or not url.strip():
        return None

    mht_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL or use ID
    filename = url.split("/")[-1] if "/" in url else f"{id_value}.mht"
    # Clean filename
    filename = filename.split("?")[0]  # Remove query params
    if not filename.endswith(".mht"):
        filename = f"{id_value}.mht"

    file_path = mht_dir / filename

    # Skip if already exists
    if file_path.exists():
        logger.debug(f"MHT file already exists: {file_path}")
        return file_path

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            response.raise_for_status()

            # Save as binary (MHT files can contain binary data)
            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded MHT file: {filename} ({len(response.content):,} bytes)")
            return file_path

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed to download MHT {url}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download MHT {url} after {retries} attempts: {e}")
                return None

    return None


def load_metadata_from_chroma(collection: chromadb.Collection) -> dict[str, MedicineMetadata]:
    """Load existing metadata from ChromaDB collection"""
    try:
        results = collection.get()
        metadata_dict: dict[str, MedicineMetadata] = {}

        for i, medicine_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            if not metadata:
                continue

            # Create MedicineMetadata from ChromaDB metadata
            medicine_metadata = MedicineMetadata(
                id=medicine_id,
                ukrainian_name=metadata.get("ukrainian_name", ""),
                international_name=metadata.get("international_name", ""),
                medicinal_product_name=metadata.get("medicinal_product_name", ""),
                file_path=metadata.get("html_file_path", ""),
                mht_file_path=metadata.get("mht_file_path", ""),
                fetch_timestamp=metadata.get("fetch_timestamp", ""),
                status=metadata.get("fetch_status", "pending"),
                error_message=metadata.get("error_message"),
            )
            metadata_dict[medicine_id] = medicine_metadata

        return metadata_dict
    except Exception as e:
        logger.warning(f"Error loading metadata from ChromaDB: {e}. Starting fresh.")
        return {}


def init_chroma_client(chroma_dir: Path, collection_name: str) -> tuple[chromadb.Client, chromadb.Collection]:
    """Initialize ChromaDB client and get/create medicines collection"""
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Medicine registry with metadata and future embeddings"},
    )

    logger.info(f"ChromaDB initialized at {chroma_dir}, collection: {collection_name}")
    return client, collection


def save_medicine_to_chroma(
    collection: chromadb.Collection,
    medicine_data: dict,
) -> None:
    """Save or update medicine data in ChromaDB (from CSV extraction)"""
    medicine_id = medicine_data["id"]
    timestamp = datetime.now().isoformat()
    instruction_url = medicine_data.get("instruction_url", "")

    # Check if document exists
    existing = collection.get(ids=[medicine_id])

    if existing["ids"]:
        # Update existing document
        collection.update(
            ids=[medicine_id],
            metadatas=[{
                "ukrainian_name": medicine_data["ukrainian_name"],
                "international_name": medicine_data["international_name"],
                "medicinal_product_name": medicine_data["medicinal_product_name"],
                "instruction_url": instruction_url,
                "updated_at": timestamp,
            }],
        )
    else:
        # Create new document with empty document text (will be filled with embeddings later)
        collection.add(
            ids=[medicine_id],
            documents=[medicine_data.get("ukrainian_name", "") or ""],  # Use name as document text for now
            metadatas=[{
                "ukrainian_name": medicine_data["ukrainian_name"],
                "international_name": medicine_data["international_name"],
                "medicinal_product_name": medicine_data["medicinal_product_name"],
                "instruction_url": instruction_url,
                "fetch_status": "pending",
                "created_at": timestamp,
                "updated_at": timestamp,
            }],
        )


def update_ingestion_status_chroma(
    collection: chromadb.Collection,
    metadata: MedicineMetadata,
) -> None:
    """Update medicine ingestion status in ChromaDB"""
    # Get existing metadata
    existing = collection.get(ids=[metadata.id])
    if not existing["ids"]:
        logger.warning(f"Medicine {metadata.id} not found in ChromaDB, skipping update")
        return

    # Merge existing metadata with new status
    existing_metadata = existing["metadatas"][0] if existing["metadatas"] else {}
    updated_metadata = {
        **existing_metadata,
        "html_file_path": metadata.file_path,
        "mht_file_path": metadata.mht_file_path,
        "fetch_status": metadata.status,
        "fetch_timestamp": metadata.fetch_timestamp,
        "error_message": metadata.error_message,
        "updated_at": datetime.now().isoformat(),
    }

    collection.update(
        ids=[metadata.id],
        metadatas=[updated_metadata],
    )


def process_medicine(
    medicine_data: dict,
    html_dir: Path,
    mht_dir: Path,
    metadata_dict: dict[str, MedicineMetadata],
    chroma_collection: Optional[chromadb.Collection] = None,
    skip_existing: bool = True,
) -> MedicineMetadata:
    """Process a single medicine: fetch HTML, download MHT, and save metadata"""
    id_value = medicine_data["id"]
    instruction_url = medicine_data.get("instruction_url", "")

    # Check if already processed
    if skip_existing and id_value in metadata_dict:
        existing = metadata_dict[id_value]
        html_exists = existing.file_path and Path(existing.file_path).exists()
        mht_exists = existing.mht_file_path and Path(existing.mht_file_path).exists()
        if existing.status == "success" and html_exists and (not instruction_url or mht_exists):
            logger.debug(f"Skipping {id_value} - already processed")
            return existing

    # Fetch HTML
    html_content = fetch_html(id_value)
    html_file_path = ""
    mht_file_path = ""

    if html_content:
        # Save HTML file
        html_file_path = str(save_html_file(id_value, html_content, html_dir))

        # Download MHT file if URL is available
        if instruction_url:
            mht_path = download_mht_file(instruction_url, id_value, mht_dir)
            if mht_path:
                mht_file_path = str(mht_path)
            else:
                logger.warning(f"Failed to download MHT file for {id_value} from {instruction_url}")

        # Create metadata
        metadata = MedicineMetadata(
            id=id_value,
            ukrainian_name=medicine_data["ukrainian_name"],
            international_name=medicine_data["international_name"],
            medicinal_product_name=medicine_data["medicinal_product_name"],
            file_path=html_file_path,
            mht_file_path=mht_file_path,
            fetch_timestamp=datetime.now().isoformat(),
            status="success",
        )
    else:
        # Failed to fetch
        metadata = MedicineMetadata(
            id=id_value,
            ukrainian_name=medicine_data["ukrainian_name"],
            international_name=medicine_data["international_name"],
            medicinal_product_name=medicine_data["medicinal_product_name"],
            file_path="",
            mht_file_path="",
            fetch_timestamp=datetime.now().isoformat(),
            status="failed",
            error_message="Failed to fetch HTML content",
        )

    metadata_dict[id_value] = metadata

    # Update ChromaDB
    if chroma_collection:
        update_ingestion_status_chroma(chroma_collection, metadata)

    return metadata


def main() -> None:
    """Main ingestion function"""
    logger.info("Starting medical instructions ingestion")

    # Validate CSV file exists
    if not CSV_PATH.exists():
        logger.error(f"CSV file not found: {CSV_PATH}")
        return

    # Read CSV
    csv_size = CSV_PATH.stat().st_size if CSV_PATH.exists() else 0
    csv_size_mb = csv_size / (1024 * 1024)
    logger.info(f"Reading CSV from {CSV_PATH}")
    logger.info(f"  File size: {csv_size_mb:.2f} MB ({csv_size:,} bytes)")
    rows = read_csv_with_encoding(CSV_PATH)
    logger.info(f"  Total rows read: {len(rows):,}")
    if LIMIT and LIMIT > 0:
        logger.info(f"  Will process up to {LIMIT} medicines (limit enabled)")
    else:
        logger.info(f"  Will process all medicines (no limit)")

    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {CHROMA_DIR}")
    chroma_client, chroma_collection = init_chroma_client(CHROMA_DIR, CHROMA_MEDICINES_COLLECTION)

    # Extract medicine data (stop when limit is reached)
    logger.info("Extracting medicine data from CSV")
    medicines: list[dict] = []
    for row in rows:
        medicine_data = extract_medicine_data(row)
        if medicine_data:
            medicines.append(medicine_data)
            # Save to ChromaDB immediately after extraction
            save_medicine_to_chroma(chroma_collection, medicine_data)

            # Stop reading if we've reached the limit
            if LIMIT and LIMIT > 0 and len(medicines) >= LIMIT:
                logger.info(f"Reached ingestion limit of {LIMIT} items, stopping CSV reading")
                break

    logger.info(f"Found {len(medicines)} medicines to process")
    logger.info(f"Saved {len(medicines)} medicines to ChromaDB")

    # Load existing metadata from ChromaDB
    metadata_dict = load_metadata_from_chroma(chroma_collection)
    logger.info(f"Loaded {len(metadata_dict)} existing metadata entries from ChromaDB")

    # Process each medicine
    logger.info("Starting HTML fetching...")
    success_count = 0
    failed_count = 0

    with tqdm(total=len(medicines), desc="Processing medicines") as pbar:
        for medicine_data in medicines:
            metadata = process_medicine(
                medicine_data,
                HTML_DIR,
                MHT_DIR,
                metadata_dict,
                chroma_collection=chroma_collection,
                skip_existing=True,
            )

            if metadata.status == "success":
                success_count += 1
            else:
                failed_count += 1

            # Metadata is automatically saved to ChromaDB in update_ingestion_status_chroma()
            # No need for periodic saves

            # Rate limiting
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

            pbar.update(1)

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"Total processed: {len(medicines)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"HTML files saved to: {HTML_DIR}")
    logger.info(f"MHT files saved to: {MHT_DIR}")
    logger.info(f"Metadata saved to ChromaDB: {CHROMA_DIR}")
    logger.info(f"Collection: {CHROMA_MEDICINES_COLLECTION}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

