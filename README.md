# Hard negatives mining pipeline

This project builds a training dataset `(query, positive, hard negatives)` for embedder and reranker training.

## Run modes

### 1) venv + pip (recommended on HPC, no Docker required)

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. Prepare `.env` (based on `.env.example.base` or `.env.example.large`).
3. Set vector DB backend:
   - LanceDB (default, no Docker):
     ```bash
     export VECTOR_DB_BACKEND=lancedb
     export LANCEDB_PATH=./lancedb_data
     ```
   - Qdrant (optional):
     ```bash
     export VECTOR_DB_BACKEND=qdrant
     export QDRANT_URL=http://localhost:6333
     ```

### 2) Docker Compose (optional local convenience setup)

```bash
docker compose up -d
```

This starts the `executable` container and optional `vdb` service based on Qdrant.

## Pipeline

1. Prepare data (`corpus.parquet`, `queries.parquet`, `relevant.parquet`):
   ```bash
   python app_code/to_huggingface_dataset.py
   # or
   python app_code/msmarco_to_huggingface_dataset.py
   # or
   python app_code/clips_mqa_to_huggingface_dataset.py
   ```
2. Load documents into vector DB:
   ```bash
   python app_code/add_documents_to_db.py
   ```
3. Add scores for positives:
   ```bash
   python app_code/add_positives_ranks.py
   ```
4. Mine negatives:
   ```bash
   python app_code/find_negatives.py
   ```
5. Build FlagEmbedding JSONL:
   ```bash
   python app_code/create_flag_embedding_jsonl.py
   ```

## Backend notes

- `VECTOR_DB_BACKEND=lancedb` uses dense retrieval + LanceDB BM25/hybrid.
- `VECTOR_DB_BACKEND=qdrant` keeps dense+sparse (SPLADE) flow.
