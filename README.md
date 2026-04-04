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

## Quick end-to-end smoke test (sample JSONL)

If you want to quickly verify that the full pipeline wiring works, you can run it on the tiny sample file already in the repo: `app_code/data/input.jsonl`.

> Note: `find_negatives.py` requires CUDA GPUs in the current implementation.

```bash
# 0) Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Use LanceDB (no Docker required)
export VECTOR_DB_BACKEND=lancedb
export LANCEDB_PATH=./.smoke/lancedb
mkdir -p ./.smoke

# 2) Build tiny parquet inputs from sample jsonl
python app_code/to_huggingface_dataset.py \
  --input_file_path app_code/data/input.jsonl \
  --queries_path ./.smoke/queries.parquet \
  --corpus_path ./.smoke/corpus.parquet \
  --relevant_path ./.smoke/relevant.parquet

# 3) Ingest corpus to vector DB (small, fast models)
python app_code/add_documents_to_db.py \
  --dataset_path ./.smoke/corpus.parquet \
  --dense_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 8 \
  --database_collection_name smoke_documents

# 4) Score positives
python app_code/add_positives_ranks.py \
  --queries_path ./.smoke/queries.parquet \
  --corpus_path ./.smoke/corpus.parquet \
  --relevant_path ./.smoke/relevant.parquet \
  --output_path ./.smoke/relevant_with_score.parquet \
  --chunk_size 64 \
  --reranker_batch_size 8 \
  --reranker_model_name cross-encoder/ms-marco-MiniLM-L-6-v2

# 5) Mine negatives
python app_code/find_negatives.py \
  --dense_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --reranker_model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --embedding_batch_size 8 \
  --reranker_batch_size 8 \
  --database_collection_name smoke_documents \
  --queries_path ./.smoke/queries.parquet \
  --relevant_path ./.smoke/relevant_with_score.parquet \
  --output_path ./.smoke/negatives.parquet \
  --top_k 20
```

Expected smoke-test artifacts:
- `./.smoke/queries.parquet`
- `./.smoke/corpus.parquet`
- `./.smoke/relevant.parquet`
- `./.smoke/relevant_with_score.parquet`
- `./.smoke/negatives.parquet`
