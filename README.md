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

**Phase 3: Web Interface** (Completed)
- Streamlit chat interface
- Query processing
- Response generation
- Multilingual support (Ukrainian questions → English/Ukrainian responses)

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

**Important:** Ingestion downloads files and saves metadata. It does NOT create embeddings - that's done by the indexing script.

Run the ingestion script to fetch medical instructions:

**Local development:**
```bash
python -m app.ingestion
```

**On production server:**
```bash
# SSH to server first
gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c

# Switch to rag-app user and activate venv
sudo -u rag-app bash
cd /opt/rag-medicine-instructions
source .venv/bin/activate

# Now run ingestion
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

**Note:** Ingestion can take a long time (hours) for all 16,767 items. Use `INGESTION_LIMIT` in `.env` to limit processing for testing.

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

**Important:** Indexing is a separate step that runs AFTER ingestion. Ingestion only downloads files and saves metadata - it does NOT create embeddings. You must run ingestion first, then indexing.

Run the indexing script to process HTML and MHT files and create embeddings:

**Local development:**
```bash
python -m app.indexing
```

**On production server:**
```bash
# SSH to server first
gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c

# Switch to rag-app user and activate venv
sudo -u rag-app bash
cd /opt/rag-medicine-instructions
source .venv/bin/activate

# Now run indexing
python -m app.indexing
```

This will:
- Read HTML and MHT files from `data/html/` and `data/mht/` (already downloaded by ingestion)
- Extract text content from both file types
- Chunk text into manageable pieces (with overlap)
- Generate embeddings using OpenAI API (requires `OPENAI_API_KEY`)
- Store chunks in ChromaDB RAG collection with metadata:
  - `medicine_id` - Links back to medicines collection
  - `source_file` - Path to original file
  - `file_type` - "html" or "mht"
  - `chunk_index` - Chunk number within the file
  - `total_chunks` - Total chunks in the file

**Complete workflow example (on server):**
```bash
# SSH to server and switch to rag-app user
gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c
sudo -u rag-app bash
cd /opt/rag-medicine-instructions
source .venv/bin/activate

# Step 1: Run ingestion to download files
python -m app.ingestion

# Step 2: Run indexing to create embeddings (after ingestion completes)
python -m app.indexing
```

**Note:** 
- Indexing requires `OPENAI_API_KEY` in `.env` file
- Indexing can take a long time and cost money (OpenAI API calls)
- Use `INDEXING_LIMIT` in `.env` to limit processing for testing
- The script shows progress, token usage, and estimated costs

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

### Web Interface (Phase 3)

Run the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Or with custom port:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

The web interface provides:
- Chat interface for querying medical instructions
- Semantic search across all indexed instructions
- Multilingual support (ask in Ukrainian, get answer in English/Ukrainian)
- Source citations with medicine information
- Response language selection (English, Ukrainian, Russian)

Access the application at `http://localhost:8501` (or configured port).

## Deployment

The application can be deployed to a production server with automated deployment via GitHub Actions.

### Server Setup

1. **Connect to the server:**
   ```bash
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c
   ```

2. **Run the setup script:**
   
   **Option A: If you're already on the server:**
   ```bash
   # Clone the repository or copy files to server first
   # Then run:
   chmod +x scripts/setup_server.sh
   sudo scripts/setup_server.sh
   ```
   
   **Option B: From your local machine:**
   ```bash
   # Copy setup script to server
   gcloud compute scp scripts/setup_server.sh rag-medicine-instructions-01:/tmp/ --zone us-east1-c
   
   # SSH to server and run
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c
   chmod +x /tmp/setup_server.sh
   sudo /tmp/setup_server.sh
   ```

3. **Copy application code:**
   
   **Option A: Clone repository on server (Recommended for production):**
   ```bash
   # On server
   sudo -u rag-app git clone <your-repo-url> /opt/rag-medicine-instructions
   sudo chown -R rag-app:rag-app /opt/rag-medicine-instructions
   ```
   
   **Option B: Create archive and copy from local machine (for manual deployment):**
   ```bash
   # On local machine, create a tar archive excluding unnecessary files
   cd /path/to/rag-medicine-instructions
   tar --exclude='.git' \
       --exclude='.venv' \
       --exclude='__pycache__' \
       --exclude='*.pyc' \
       --exclude='data' \
       --exclude='storage' \
       --exclude='.env' \
       --exclude='.DS_Store' \
       --exclude='*.ipynb_checkpoints' \
       --exclude='.cursor' \
       -czf /tmp/rag-app.tar.gz .
   
   # Copy archive to server
   gcloud compute scp /tmp/rag-app.tar.gz rag-medicine-instructions-01:/tmp/ --zone us-east1-c
   
   # Extract archive on server
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c --command="sudo -u rag-app mkdir -p /opt/rag-medicine-instructions && sudo -u rag-app tar -xzf /tmp/rag-app.tar.gz -C /opt/rag-medicine-instructions && sudo chown -R rag-app:rag-app /opt/rag-medicine-instructions && rm /tmp/rag-app.tar.gz"
   
   # Clean up local archive
   rm /tmp/rag-app.tar.gz
   
   # Remove unrelated files from server (notebooks, documentation, etc.)
   # Note: reestr.csv is needed for ingestion, so don't remove it!
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c --command="cd /opt/rag-medicine-instructions && sudo -u rag-app rm -f explore_chromadb.ipynb search_rag.ipynb STORAGE_ANALYSIS.md && sudo -u rag-app find . -name '._*' -type f -delete"
   ```
   
   **Option C: Manual file copy (for small changes):**
   ```bash
   # Copy specific files/directories using gcloud compute scp
   gcloud compute scp --recurse app/ rag-medicine-instructions-01:/opt/rag-medicine-instructions/ --zone us-east1-c
   gcloud compute scp --recurse scripts/ rag-medicine-instructions-01:/opt/rag-medicine-instructions/ --zone us-east1-c
   gcloud compute scp --recurse systemd/ rag-medicine-instructions-01:/opt/rag-medicine-instructions/ --zone us-east1-c
   gcloud compute scp --recurse nginx/ rag-medicine-instructions-01:/opt/rag-medicine-instructions/ --zone us-east1-c
   gcloud compute scp requirements.txt rag-medicine-instructions-01:/opt/rag-medicine-instructions/ --zone us-east1-c
   ```
   
   **Note:** `gcloud compute scp` doesn't support `--exclude` option. For automated deployments, use GitHub Actions (see below) or `git clone` on the server.

4. **Copy required files (.env and reestr.csv):**
   ```bash
   # Copy .env file
   gcloud compute scp .env rag-medicine-instructions-01:/tmp/.env --zone us-east1-c
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c --command="sudo mv /tmp/.env /opt/rag-medicine-instructions/.env && sudo chown rag-app:rag-app /opt/rag-medicine-instructions/.env && sudo chmod 600 /opt/rag-medicine-instructions/.env"
   
   # Copy reestr.csv (required for ingestion)
   gcloud compute scp reestr.csv rag-medicine-instructions-01:/tmp/reestr.csv --zone us-east1-c
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c --command="sudo mv /tmp/reestr.csv /opt/rag-medicine-instructions/reestr.csv && sudo chown rag-app:rag-app /opt/rag-medicine-instructions/reestr.csv"
   ```

5. **Set up directory permissions:**
   ```bash
   # Ensure rag-app user can write to data and storage directories
   gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c --command="sudo chown -R rag-app:rag-app /opt/rag-medicine-instructions && sudo chmod -R u+w /opt/rag-medicine-instructions && sudo -u rag-app mkdir -p /opt/rag-medicine-instructions/data/html /opt/rag-medicine-instructions/data/mht /opt/rag-medicine-instructions/storage/chroma"
   ```

6. **Install Python dependencies:**
   ```bash
   # On server
   sudo -u rag-app bash
   cd /opt/rag-medicine-instructions
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. **Install systemd service:**
   ```bash
   # On server
   sudo cp systemd/rag-medicine-instructions.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable rag-medicine-instructions
   sudo systemctl start rag-medicine-instructions
   ```

8. **Configure Nginx:**
   ```bash
   # On server
   sudo cp nginx/rag-medicine-instructions.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/rag-medicine-instructions.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

9. **Set up SSL certificate:**
   ```bash
   # On server
   sudo certbot --nginx -d rag-medicine-instructions.medkit.space
   ```

### GitHub Actions Deployment

The project includes automated deployment via GitHub Actions. When code is pushed or merged to the `main` branch, it will automatically:

1. Stop the service
2. Deploy new code
3. Update dependencies
4. Restart the service

**Required GitHub Secrets:**
- `SSH_PRIVATE_KEY` - Private SSH key for server access
- `SSH_HOST` - Server IP address (e.g., `34.73.183.90`)
- `SSH_USER` - SSH username for server

**To set up GitHub Secrets:**
1. Go to repository Settings → Secrets and variables → Actions
2. Add the three secrets listed above

**Manual Deployment:**

If you need to deploy manually:

```bash
# On server
sudo -u rag-app bash
cd /opt/rag-medicine-instructions
git pull  # or use rsync
source .venv/bin/activate
pip install -r requirements.txt
exit  # exit from rag-app shell
sudo systemctl restart rag-medicine-instructions
```

Or use the deployment script:

```bash
# On server
sudo /opt/rag-medicine-instructions/scripts/deploy.sh
```

### Monitoring

**View service logs:**
```bash
sudo journalctl -u rag-medicine-instructions -f
```

**Check service status:**
```bash
sudo systemctl status rag-medicine-instructions
```

**View Nginx logs:**
```bash
sudo tail -f /var/log/nginx/rag-medicine-instructions-*.log
```

### Production URL

- **Domain:** https://rag-medicine-instructions.medkit.space
- **IP:** 34.73.183.90

## Project Structure

```
rag-medicine-instructions/
├── app/
│   ├── __init__.py
│   ├── ingestion.py          # Ingestion script
│   ├── indexing.py            # Indexing script
│   └── streamlit_app.py      # Streamlit web application
├── scripts/
│   ├── setup_server.sh        # Server setup script
│   └── deploy.sh              # Manual deployment script
├── systemd/
│   └── rag-medicine-instructions.service  # Systemd service file
├── nginx/
│   └── rag-medicine-instructions.conf    # Nginx configuration
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions deployment workflow
├── data/
│   ├── html/                 # Downloaded HTML instruction files
│   └── mht/                  # Downloaded MHT instruction files
├── storage/
│   └── chroma/               # ChromaDB storage (medicine metadata + embeddings)
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

## Notes

- **ChromaDB Management**: ChromaDB runs as an embedded database within the Python process. When the Streamlit service stops, ChromaDB automatically closes connections. No separate service is needed for ChromaDB.
- **Environment Variables**: All configuration is done via environment variables in `.env` file. See `.env.example` for all available options.
- **Backup**: The deployment script automatically creates backups of ChromaDB before updates. Backups are stored in `/opt/rag-medicine-instructions/backups/`.

