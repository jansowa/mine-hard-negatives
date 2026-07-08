# Curated Negatives Tools

Tools in this directory operate on negatives mined outside this project and normalise them to the
FlagEmbedding JSONL shape:

```json
{"query": "...", "pos": ["..."], "neg": ["..."], "pos_scores": [1.0], "neg_scores": [0.2], "prompt": "", "type": "retrieval"}
```

## Quickest way to run LightOn mining

From the project root, create and activate a virtual environment, install dependencies, and prepare a local
`.env` file:

```bash
uv venv --python 3.12.3 .venv
source .venv/bin/activate
uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements-lancedb.txt
cp .env.example.lightonai.mxbai .env
```

In `.env`, select the LightOn splits to process. This is the main variable that normally needs to be changed:

```dotenv
LIGHTONAI_SPLITS="fiqa"
```

Multiple splits can be provided as a comma-separated list:

```dotenv
LIGHTONAI_SPLITS="fiqa,trivia,fever"
```

Available values are: `fiqa`, `trivia`, `hotpotqa`, `nq`, `msmarco`, `fever`, and `squadv2`.

Run the complete pipeline:

```bash
python app_code/curated_negatives/run_lightonai_adaptive_pipeline.py
```

Each split is written to its own `data/lightonai_pipeline/<split>/` directory. Running the same command again
resumes processing and reuses existing artifacts and scores.

## Recommended LightOn adaptive pipeline

The LightOn dataset card describes three configs: `queries`, `documents`, and `scores`. In `scores`, the
first `document_id` is the positive document, followed by mined negatives and their original scores.
The loader passes split-scoped `data_files` such as `scores/fiqa-*`, `queries/fiqa-*`, and
`documents/fiqa-*`, so selecting `LIGHTONAI_SPLITS="fiqa,trivia,fever"` does not download the large
`msmarco` shards.

Start by creating a local `.env` from the shareable example:

```bash
cp .env.example.lightonai.mxbai .env
```

The example config uses one final reranker for this whole flow:

```dotenv
RERANKER_NAME="mixedbread-ai/mxbai-rerank-base-v2"
CANDIDATE_RERANKER_NAME="mixedbread-ai/mxbai-rerank-base-v2"
FINAL_RERANKER_NAME="mixedbread-ai/mxbai-rerank-base-v2"
RERANKER_MAX_LENGTH=512
BETA=0.01
U_FLOOR=0.005
FINAL_BETA=0.01
FINAL_U_FLOOR=0.005
FINAL_RERANK_MAX_BUDGET=0
NUM_NEGATIVES=10
LIGHTONAI_DATASET_NAME="lightonai/embeddings-fine-tuning"
LIGHTONAI_SPLITS="fiqa"
LIGHTONAI_PIPELINE_ROOT="data/lightonai_pipeline"
LIGHTONAI_STAGES="artifacts,positives,negatives,jsonl"
LIGHTONAI_REBUILD_ARTIFACTS=false
```

`FINAL_RERANK_MAX_BUDGET=0` means adaptive final reranking may inspect all LightOn candidates for a query
until it finds `NUM_NEGATIVES` strict negatives or exhausts that query's candidate list.

For larger LightOn splits, avoid an intermediate JSONL with repeated texts. Run the compact adaptive pipeline:

```bash
python app_code/curated_negatives/run_lightonai_adaptive_pipeline.py
```

The runner derives paths from the split name. For `LIGHTONAI_SPLITS="fiqa"`, artifacts land in:

```text
data/lightonai_pipeline/fiqa/queries.parquet
data/lightonai_pipeline/fiqa/corpus.parquet
data/lightonai_pipeline/fiqa/relevant.parquet
data/lightonai_pipeline/fiqa/negative_candidates.parquet
data/lightonai_pipeline/fiqa/relevant_with_score.parquet
data/lightonai_pipeline/fiqa/negatives.parquet
data/lightonai_pipeline/fiqa/train.jsonl
```

To run several splits, edit your local `.env`:

```dotenv
LIGHTONAI_SPLITS="fiqa,nq,msmarco"
```

and run the same command again. Each split gets its own folder under `LIGHTONAI_PIPELINE_ROOT`.
Complete existing compact artifacts are reused automatically. Set
`LIGHTONAI_REBUILD_ARTIFACTS=true` only when a split must be downloaded and rebuilt intentionally.

You can also run a subset of stages, for example after artifacts already exist:

```bash
python app_code/curated_negatives/run_lightonai_adaptive_pipeline.py --stages positives,negatives,jsonl
```

Original LightOn scores are preserved under `original_pos_scores` and `original_neg_scores`; final reranker
scores remain the main `pos_scores` and `neg_scores`.

### Smoke runs, resume, and synthetic positives

Use a deterministic query window for a quick LightOn experiment:

```dotenv
PIPELINE_SAMPLE_SKIP=0
PIPELINE_SAMPLE_LIMIT=100
```

Later set `PIPELINE_SAMPLE_LIMIT=0` and run the same command. Positive and candidate score artifacts keep
completed query-document pairs, so the reranker scores only missing pairs. The final JSONL is rebuilt for the
currently selected query window.

If final reranking was interrupted after scores reached `negatives_worker_0_0.jsonl` but before
`negatives.parquet` was created, running only the `jsonl` stage automatically consolidates that worker file.
This does not load the reranker or calculate new scores.

Synthetic-positive mining runs only while exporting JSONL. It considers candidates already scored by the final
reranker; it never expands adaptive reranking and never invokes a model:

```dotenv
MINE_POSITIVES=true
MAX_MINED_POSITIVES=1
U_SANITY_CEILING=0.90
U_ABSOLUTE_CEILING=0.995
U_POSITIVE_BETA=0.95
POSITIVE_NEAR_DUPLICATE_THRESHOLD=0.80
```

All `U_*` thresholds operate on percentiles of final-reranker scores for organic positives. A candidate must pass
the sanity ceiling and either the absolute ceiling or the beta-relative threshold. Candidates with at least 80%
word-token overlap with any organic or already selected synthetic positive are rejected. Every exported row
contains `pos_is_synthetic`, aligned with `pos`, `pos_id`, and `pos_scores`.

For the older name used in LightOn's sample code, pass:

```bash
python app_code/curated_negatives/run_lightonai_adaptive_pipeline.py \
  --dataset_name lightonai/nv-embed-supervised-distill-dedup
```

## Direct LightOn JSONL export

For smaller experiments, you can still export directly to FlagEmbedding JSONL with NV-style filtering:

```bash
python app_code/curated_negatives/lightonai_to_flag_embedding.py \
  --dataset_name lightonai/embeddings-fine-tuning \
  --num_negatives 50 \
  --nv_threshold 0.99 \
  --output_dir data/lightonai_flag_embedding
```

## Translate a LightOn HotpotQA JSONL to Polish

The integer LightOn HotpotQA query and document IDs are zero-based line indexes into the corresponding
BEIR/CLARIN files. The translator uses that exact mapping to replace every `query`, `pos`, and `neg` text while
preserving IDs, scores, and all other fields:

```bash
python app_code/curated_negatives/translate_lightonai_beir_pl_jsonl.py \
  --input_path data/lightonai_pipeline/hotpotqa/train.jsonl
```

This creates `data/lightonai_pipeline/hotpotqa/train_pl.jsonl`. Missing source files are downloaded from
`clarin-knext/hotpotqa-pl` to `data/hotpotqa_pl_source`. The conversion fails instead of producing a partially
translated file if any required query or document index is unavailable.

The same line-index mapping is available for other corresponding LightOn and CLARIN BEIR-PL datasets. For FIQA:

```bash
python app_code/curated_negatives/translate_lightonai_beir_pl_jsonl.py \
  --input_path data/lightonai_pipeline/fiqa/train.jsonl \
  --source_dataset clarin-knext/fiqa-pl \
  --source_dir data/fiqa_pl_source
```

## Rescore an existing FlagEmbedding JSONL

This computes fresh reranker scores for positives and negatives. Existing `pos_scores` and `neg_scores`
are copied to `original_pos_scores` and `original_neg_scores`; new reranker scores become the main
FlagEmbedding-compatible `pos_scores` and `neg_scores`.

```bash
python app_code/curated_negatives/score_flag_embedding_jsonl.py \
  --input_path data/lightonai_flag_embedding/nq.jsonl \
  --output_path data/lightonai_flag_embedding/nq.reranked.jsonl \
  --reranker_model_name BAAI/bge-reranker-v2-m3 \
  --reranker_batch_size 16 \
  --record_batch_size 32 \
  --resume
```

The scorer writes to `OUTPUT.incomplete` and replaces `OUTPUT` only after finishing. If the process is
interrupted, run the same command again with `--resume`; already written rows are counted and skipped.
