# Hard negatives mining pipeline

This project builds a training dataset `(query, positive, hard negatives)` for embedder and reranker training.

## Run modes

### 1) venv + pip (recommended on HPC, no Docker required)

1. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.12.3 .venv
   source .venv/bin/activate
   uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements-lancedb.txt
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
     uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements.txt
     ```

If installation fails while building old `ir-datasets` dependencies such as `cbor`,
make sure the system has a C build toolchain, then clear the failed build cache:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev
uv cache clean cbor warc3-wet-clueweb09 flagembedding setuptools
uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements-lancedb.txt
```

### 2) Docker Compose (optional local convenience setup)

```bash
docker compose up -d
```

This starts the `executable` container and optional `vdb` service based on Qdrant.

## Updating Requirements

The runtime `.in` files are the source of truth. Compile pinned requirements with `uv`:

```bash
uv pip compile requirements-lancedb.in -o requirements-lancedb.txt \
  --python-version 3.12.3 \
  --torch-backend cu128 \
  --build-constraint build-constraints.txt \
  --emit-index-url \
  --emit-find-links

uv pip compile requirements.in -o requirements.txt \
  --python-version 3.12.3 \
  --torch-backend cu128 \
  --build-constraint build-constraints.txt \
  --emit-index-url \
  --emit-find-links
```

GitHub CI uses `requirements-ci.txt`, maintained alongside the deliberately lightweight `requirements-ci.in`
direct dependency set for `make check`. Keep runtime mining dependencies such as PyTorch, LanceDB/Qdrant,
Transformers, and FlagEmbedding in the production requirement files unless a unit test truly needs them.

## Code quality

Run Ruff checks:

```bash
ruff check app_code tests
```

Run mypy type checks:

```bash
mypy
```

Apply safe automatic fixes:

```bash
ruff check app_code tests --fix
```

Format Python files:

```bash
ruff format app_code tests
```

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

## Optional two-stage negative reranking

For cheaper mining, you can split negative scoring into a small-reranker candidate pass and a final large-reranker pass. The intermediate candidate artifact is intentionally light: it stores IDs, retrieval metadata, and candidate scores, but not duplicated query/document texts. A complete example configuration is available in `.env.example.sdadas.two-stage`; with that file copied to `.env`, the commands below do not need path/model flags.

### 1) Workstation: score candidates with a small reranker

First score positives with the same small reranker so the candidate percentile thresholds use the matching score distribution:

```bash
python app_code/add_positives_ranks.py
```

Then mine candidate negatives and save the small-reranker scores as `candidate_ranking`.
The candidate-stage thresholds come from `CANDIDATE_BETA` and `CANDIDATE_U_FLOOR`, falling back to `BETA`
and `U_FLOOR`. The old `TOP_K` option is kept for command compatibility, but iterative mining is now controlled
by `CANDIDATE_TARGET`, `CANDIDATE_SEARCH_CHUNK`, `CANDIDATE_MAX_OFFSET_ITERS`, and `CANDIDATE_RANDOM_FALLBACK`.

```bash
python app_code/find_negatives.py
```

Upload or otherwise move these files together:

- `data/queries.parquet`
- `data/corpus.parquet`
- `data/negative_candidates.parquet`
- the relevant file needed by your next stage, for example `data/relevant_with_candidate_score.parquet`

### 2) Larger machine: score positives with the final reranker

Adaptive final reranking uses the final reranker's positive-score distribution for thresholds, so score positives
with the final model before reranking negative candidates:

```bash
python app_code/add_positives_ranks.py --final-step
```

### 3) Larger machine: adaptive final reranking

The final pass uses `FINAL_RERANK_MODE=adaptive` by default. It starts with the best small-reranker candidates,
scores only the initial budget with the large reranker, and expands the per-query budget only when the final
threshold still yields fewer than `NUM_NEGATIVES` strict negatives. It writes `final_ranking`, a compatibility
`ranking` alias, final-selection metadata, and a JSON report.

```bash
python app_code/rerank_negative_candidates.py
```

Finally build JSONL using the final negative score column:

```bash
python app_code/create_flag_embedding_jsonl.py
```

The JSONL builder first chooses strict final negatives. With `BACKFILL_POLICY=relaxed`, it fills any missing
slots from the safest final-scored relaxed candidates instead of silently producing very short negative lists.
It also writes an export report with the final negatives-per-query histogram.

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
uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements-lancedb.txt

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
  --candidate_target 20

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
