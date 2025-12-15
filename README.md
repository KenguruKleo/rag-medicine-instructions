# Medical Instructions RAG System

A RAG (Retrieval-Augmented Generation) system for Ukrainian medical instructions, built with LlamaIndex, ChromaDB, and Streamlit.

## Project Status

**Phase 1: Ingestion** (Current)
- CSV registry processing
- ChromaDB storage for medicine data and metadata
- HTML instruction fetching
- All metadata stored in ChromaDB (no JSON files)

**Phase 2: Indexing** (Current)
- HTML and MHT file parsing
- Text extraction and chunking
- OpenAI embeddings generation
- ChromaDB RAG collection with metadata (medicine_id, source_file, chunk_index)
- Semantic search capabilities

**Phase 3: Web Interface** (Planned)
- Streamlit chat interface
- Query processing
- Response generation

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment template:
```bash
cp .env.example .env
```

4. Edit `.env` if needed (optional configuration)

The `.env.example` file includes all configuration options with default values. Key parameters:
- `INGESTION_LIMIT=100` - Limits processing to 100 items for testing (set to `0` to process all)
- `REQUEST_DELAY=0.5` - Delay between HTTP requests
- Other paths and ChromaDB settings

## Usage

### Ingestion

Run the ingestion script to fetch medical instructions:

```bash
python -m app.ingestion
```

This will:
- Read `reestr.csv` (Windows-1251 encoding)
- Extract medicine IDs, names, and instruction URLs
- Save extracted data to ChromaDB collection (`storage/chroma`)
- Fetch HTML instruction pages from the Ukrainian registry
- Save HTML files to `data/html/`
- Download MHT instruction files from URLs in CSV (if available)
- Save MHT files to `data/mht/`
- Store all metadata in ChromaDB (medicine info, fetch status, file paths, timestamps)

#### Configuration Options

You can configure the ingestion process via environment variables (set in `.env` file or export before running):

- `INGESTION_LIMIT` - Limit the number of items to process (default: `100` for testing). Set to `0` or unset to process all items.
- `REQUEST_DELAY` - Delay in seconds between HTTP requests (default: `0.5`)
- `CSV_PATH` - Path to the CSV file (default: `reestr.csv`)
- `HTML_DIR` - Directory for downloaded HTML files (default: `data/html`)
- `MHT_DIR` - Directory for downloaded MHT instruction files (default: `data/mht`)
- `CHROMA_DIR` - Directory for ChromaDB storage (default: `storage/chroma`)
- `CHROMA_MEDICINES_COLLECTION` - ChromaDB collection name for medicine metadata (default: `medicines`)
- `CHROMA_RAG_COLLECTION` - ChromaDB collection name for RAG document chunks (default: `instruction_chunks`, used in Phase 2)

Example for processing all items:
```bash
INGESTION_LIMIT=0 python -m app.ingestion
```

Example with custom delay:
```bash
REQUEST_DELAY=1.0 python -m app.ingestion
```

### Exploring ChromaDB

Use the Jupyter notebook to explore ChromaDB interactively:

```bash
jupyter notebook explore_chromadb.ipynb
```

Or with JupyterLab:
```bash
jupyter lab explore_chromadb.ipynb
```

The notebook includes:
- Collection overview and statistics
- Sample document inspection
- DataFrame creation for easy analysis
- Search and filter examples

### Indexing (Phase 2)

Run the indexing script to process HTML and MHT files and create embeddings:

```bash
python -m app.indexing
```

This will:
- Read HTML and MHT files from `data/html/` and `data/mht/`
- Extract text content from both file types
- Chunk text into manageable pieces (with overlap)
- Generate embeddings using OpenAI
- Store chunks in ChromaDB RAG collection with metadata:
  - `medicine_id` - Links back to medicines collection
  - `source_file` - Path to original file
  - `file_type` - "html" or "mht"
  - `chunk_index` - Chunk number within the file
  - `total_chunks` - Total chunks in the file

#### Indexing Configuration

- `CHUNK_SIZE` - Size of text chunks (default: `1000` characters)
- `CHUNK_OVERLAP` - Overlap between chunks (default: `200` characters)
- `INDEXING_LIMIT` - Limit number of files to index (default: `0` = all)
- `OPENAI_API_KEY` - Required for embeddings
- `OPENAI_EMBED_MODEL` - Embedding model (default: `text-embedding-3-small`)

### Search (Phase 2)

Use the search notebook to query indexed instructions:

```bash
jupyter notebook search_rag.ipynb
```

The notebook includes:
- Semantic search examples (Ukrainian and English queries)
- Medicine information lookup from search results
- Results displayed with metadata (medicine_id, source file, chunk info)
- DataFrame visualization of search results
- **Multilingual RAG**: Ask questions in Ukrainian and get answers in English (or any language)

### Multilingual Capabilities

GPT-4o-mini is multilingual and can:
- ✅ Understand Ukrainian medical instructions
- ✅ Answer questions in English (or any language you specify)
- ✅ Translate and explain medical information across languages

**Example:** Ask "Які побічні ефекти?" (Ukrainian) → Get answer in English explaining the side effects from Ukrainian instructions.

See the "Multilingual RAG" section in `search_rag.ipynb` for working examples.

## Project Structure

```
rag-medicine-instructions/
├── app/
│   ├── __init__.py
│   └── ingestion.py          # Ingestion script
├── data/
│   ├── html/                 # Downloaded HTML instruction files
│   └── mht/                  # Downloaded MHT instruction files
├── storage/
│   └── chroma/               # ChromaDB storage (medicine metadata + future embeddings)
├── explore_chromadb.ipynb    # Jupyter notebook for exploring ChromaDB
├── search_rag.ipynb          # Jupyter notebook for RAG search examples
├── reestr.csv                 # Source CSV registry
├── requirements.txt
├── .env
└── README.md
```

## ChromaDB Storage

ChromaDB uses two collections for different purposes:

1. **`medicines` collection** - Stores medicine metadata (Phase 1)
2. **`instruction_chunks` collection** - Stores document chunks with embeddings for RAG (Phase 2)

### Medicines Collection Structure

The `medicines` collection stores documents with the following metadata:

- `id` - Medicine ID from CSV (used as document ID)
- `ukrainian_name` - Ukrainian trade name
- `international_name` - International non-proprietary name
- `medicinal_product_name` - Medicinal product name
- `html_file_path` - Path to downloaded HTML file
- `fetch_status` - Status: "success", "failed", or "pending"
- `fetch_timestamp` - ISO timestamp of fetch attempt
- `error_message` - Error message if fetch failed
- `created_at` - Record creation timestamp
- `updated_at` - Last update timestamp

### Querying ChromaDB

You can query ChromaDB using Python:

```python
import chromadb

client = chromadb.PersistentClient(path="storage/chroma")
collection = client.get_collection("medicines")

# Get all medicines
results = collection.get()
print(f"Total medicines: {len(results['ids'])}")

# Get medicines by status
results = collection.get(
    where={"fetch_status": "success"},
    limit=10
)

# Get specific medicine
medicine = collection.get(ids=["MEDICINE_ID"])
```

### Instruction Chunks Collection (Phase 2)

The `instruction_chunks` collection stores:
- Document chunks from HTML and MHT instruction files
- Embeddings for semantic search (1536 dimensions with text-embedding-3-small)
- Metadata for each chunk:
  - `medicine_id` - Medicine ID (links to medicines collection)
  - `source_file` - Path to original HTML/MHT file
  - `file_type` - "html" or "mht"
  - `chunk_index` - Chunk number (0-based)
  - `total_chunks` - Total number of chunks in the source file

This allows you to:
1. Search semantically across all instruction content
2. Find which medicine and file a chunk came from
3. Retrieve the full context by reading the original file

## Storage Size Analysis

See [STORAGE_ANALYSIS.md](STORAGE_ANALYSIS.md) for detailed storage size information.

**Quick Summary:**
- **Current (100 items)**: ~57 MB total
- **Estimated (all 16,767 items)**: ~9.3 GB total
  - HTML files: ~168 MB
  - MHT files: ~4.4 GB
  - ChromaDB: ~4.7 GB

## Data Source

Medical instructions are fetched from:
- Registry: http://www.drlz.com.ua/ibp/ddsite.nsf/all/shlist?opendocument
- CSV: http://www.drlz.com.ua/ibp/zvity.nsf/all/zvit/$file/reestr.csv

