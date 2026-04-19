# Hard negatives mining pipeline

This project builds a training dataset `(query, positive, hard negatives)` for embedder and reranker training.

## Run modes

### 1) venv + pip (recommended on HPC, no Docker required)

1. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.12.3 .venv
   source .venv/bin/activate
   uv pip install -r requirements-lancedb.txt
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
     Install dependencies with:
     ```bash
     uv pip install -r requirements.txt
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
   # or
   python app_code/natural_questions_to_huggingface_dataset.py
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

### LanceDB ingest notes

For LanceDB, `add_documents_to_db.py` keeps the embedding batch size separate from the database write batch size. Use `--batch_size` for model/GPU microbatching and `--db_write_batch_size` for larger LanceDB appends. The default LanceDB write batch is `4096`, which avoids creating many tiny Lance table versions.

If `--batch_size` is omitted for LanceDB, the script runs a short startup benchmark and selects the embedding microbatch size automatically. Tune the search range with `--auto_batch_size_min`, `--auto_batch_size_max`, `--auto_batch_size_candidates`, and `--auto_batch_size_sample_size`. This does not change the LanceDB append size: writes still use `--db_write_batch_size`.

LanceDB ingest overlaps embedding with database writes by default (`LANCEDB_ASYNC_WRITE=true`). The writer flushes one large completed append while the GPU starts embedding the next large chunk. Disable this with `--no_lancedb_async_write` if debugging write failures.

Document ingest resumes by default. On startup the script scans existing `document_id` values and only embeds missing documents. Pass `--no-resume` only when intentionally rebuilding a fresh table.

If an older LanceDB table was created with many tiny appends, run one ingest with `--compact_existing rebuild` after stopping any active writer. This rebuilds the current table into a compact table and leaves the old table directory as a timestamped backup.

### Negative mining performance notes

`find_negatives.py` writes worker JSONL files incrementally for crash-safe resume, then streams those files into the final Parquet output. Tune the Parquet consolidation chunk with `NEGATIVES_PARQUET_ROW_GROUP_SIZE` if needed; the default is `100000` rows.

If `--embedding_batch_size` or `--reranker_batch_size` is omitted, `find_negatives.py` auto-tunes that batch size at startup. Explicit CLI values skip startup tuning. The embedder and reranker are tuned separately with `--auto_embedding_batch_size_*` and `--auto_reranker_batch_size_*` options; both tuners stop growing candidates when CUDA memory headroom looks too small, and reranking retries with a smaller batch if a recoverable CUDA OOM still happens during mining.

For timing diagnostics, run mining with `--profile-timing` or set `NEGATIVE_PROFILE_TIMING=true`. This prints aggregate hot-path timings for search, reranking, JSONL writes, random fallback sampling, and Parquet writes. Leave profiling disabled for final throughput measurements.

With LanceDB, query embeddings are cached during negative mining to avoid recomputing the same query vector across offset groups. Control this with `LANCEDB_QUERY_VECTOR_CACHE_SIZE`; set it to `0` to disable the cache. LanceDB random fallback sampling also caches the table rows after the first fallback sample in a run.

## Quick end-to-end smoke test (sample JSONL)

If you want to quickly verify that the full pipeline wiring works, you can run it on the tiny sample file already in the repo: `app_code/data/input.jsonl`.

> Note: `find_negatives.py` requires CUDA GPUs in the current implementation.

```bash
# 0) Environment
uv venv --python 3.12.3 .venv
source .venv/bin/activate
uv pip install -r requirements-lancedb.txt

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
  --dense_model_name sdadas/mmlw-retrieval-roberta-base \
  --batch_size 8 \
  --db_write_batch_size 4096 \
  --resume \
  --database_collection_name smoke_documents

# 4) Score positives
python app_code/add_positives_ranks.py \
  --queries_path ./.smoke/queries.parquet \
  --corpus_path ./.smoke/corpus.parquet \
  --relevant_path ./.smoke/relevant.parquet \
  --output_path ./.smoke/relevant_with_score.parquet \
  --chunk_size 64 \
  --reranker_batch_size 8 \
  --reranker_model_name sdadas/polish-reranker-base-ranknet

# 5) Mine negatives
python app_code/find_negatives.py \
  --dense_model_name sdadas/mmlw-retrieval-roberta-base \
  --reranker_model_name sdadas/polish-reranker-base-ranknet \
  --embedding_batch_size 8 \
  --reranker_batch_size 8 \
  --database_collection_name smoke_documents \
  --queries_path ./.smoke/queries.parquet \
  --relevant_path ./.smoke/relevant_with_score.parquet \
  --output_path ./.smoke/negatives.parquet \
  --top_k 20

# 6) Build FlagEmbedding JSONL
python app_code/create_flag_embedding_jsonl.py \
  --corpus_path ./.smoke/corpus.parquet \
  --queries_path ./.smoke/queries.parquet \
  --relevant_path ./.smoke/relevant_with_score.parquet \
  --negatives_path ./.smoke/negatives.parquet \
  --output_path ./.smoke/train.jsonl \
  --num_negatives 5 \
  --corpus_sqlite_path ./.smoke/corpus.sqlite \
  --negcount_sqlite_path ./.smoke/negcount.sqlite \
  --query_chunk_size 64 \
  --oversample_factor 5
```

Expected smoke-test artifacts:
- `./.smoke/queries.parquet`
- `./.smoke/corpus.parquet`
- `./.smoke/relevant.parquet`
- `./.smoke/relevant_with_score.parquet`
- `./.smoke/negatives.parquet`
