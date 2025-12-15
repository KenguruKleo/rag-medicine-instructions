# Storage Size Analysis

## Current Status (100 items processed)

### File Storage
- **HTML files**: 1.00 MB (100 files)
  - Average per item: 10.25 KB
- **MHT files**: 27.08 MB (28 files)
  - Average per item: 277.34 KB
  - Note: Not all items have MHT files (only 28 out of 100)

### ChromaDB Storage
- **Total ChromaDB size**: 28.88 MB
  - Average per item: 295.69 KB
  - Includes:
    - `medicines` collection: metadata for 100 medicines
    - `instruction_chunks` collection: RAG chunks with embeddings (if indexed)

### Total Current Size
- **Combined**: 56.96 MB
- **Average per item**: 583.29 KB

---

## Estimated Size for All Items (16,767 items)

Based on current averages from 100 items:

### File Storage Estimates
- **HTML files**: ~167.84 MB
  - Calculation: 10.25 KB × 16,767 = 171.9 MB
- **MHT files**: ~4.43 GB
  - Calculation: 277.34 KB × 16,767 = 4.65 GB
  - Note: Assuming ~28% of items have MHT files (based on current 28/100 ratio)

### ChromaDB Storage Estimates
- **Estimated ChromaDB size**: ~4.73 GB
  - Calculation: 295.69 KB × 16,767 = 4.96 GB
  - Includes:
    - `medicines` collection: ~16,767 medicine metadata entries
    - `instruction_chunks` collection: RAG chunks with embeddings
    - Embeddings are the largest component (1536 dimensions × float32 = ~6KB per chunk)

### Total Estimated Size
- **Combined**: ~9.33 GB
- **Multiplier**: 167.7× current size

---

## Breakdown by Component

| Component | Current (100 items) | Estimated (16,767 items) | % of Total |
|-----------|---------------------|--------------------------|------------|
| HTML files | 1.00 MB | 167.84 MB | 1.8% |
| MHT files | 27.08 MB | 4.43 GB | 47.5% |
| ChromaDB | 28.88 MB | 4.73 GB | 50.7% |
| **Total** | **56.96 MB** | **~9.33 GB** | **100%** |

---

## Notes

1. **MHT files** are the largest component (~47.5% of total size)
   - These contain complete instruction documents with embedded resources
   - Not all medicines have MHT files (only ~28% based on current data)

2. **ChromaDB** size includes:
   - Medicine metadata (relatively small)
   - RAG chunks with embeddings (largest component)
   - Each embedding vector: 1536 dimensions × 4 bytes = ~6 KB per chunk
   - Average chunks per medicine: ~12 chunks (based on 1,203 chunks / 100 items)

3. **Storage growth**:
   - Linear scaling expected (no significant overhead)
   - ChromaDB uses efficient storage with compression

4. **Disk space recommendations**:
   - Minimum: 10 GB free space
   - Recommended: 15 GB free space (for safety margin)

---

## Cost Estimates (OpenAI API)

Based on current indexing (if all items are indexed):

- **Embedding model**: `text-embedding-3-small`
- **Price**: $0.02 per 1M tokens
- **Estimated tokens per item**: ~2,000 tokens (based on current data)
- **Total tokens for 16,767 items**: ~33.5M tokens
- **Estimated embedding cost**: ~$0.67

---

*Last updated: Based on 100 items processed*
*Total items in reestr.csv: 16,767*

