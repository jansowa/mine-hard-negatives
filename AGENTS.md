# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `app_code/`. Top-level modules implement dataset conversion, vector DB loading, scoring, negative mining, reranking, and FlagEmbedding JSONL export. `app_code/curated_negatives/` contains LightOn/BEIR import, translation, scoring, and adaptive pipeline tools. Shared vector DB helpers are in `app_code/utils/`.

Tests live in `tests/` and mirror covered modules, for example `tests/test_rerank_negative_candidates.py`. Data and generated artifacts are under `data/` and `lancedb_data/`; treat these as large local outputs unless a change explicitly updates them. Node helper scripts for JSONL post-processing live in `scripts/`.

## Build, Test, and Development Commands

- `uv venv --python 3.12.3 .venv`: create the recommended Python environment.
- `uv pip install --torch-backend cu128 --build-constraint build-constraints.txt -r requirements-lancedb.txt`: install the default LanceDB-oriented runtime dependencies.
- `make test`: run the pytest suite.
- `make lint`: run `ruff check app_code tests`.
- `make typecheck`: run `mypy`.
- `make check`: run lint, type checks, and tests.
- `make format`: format Python files with Ruff.
- `docker compose up -d`: optional local service setup, including Qdrant when using that backend.

Run pipeline commands from the repository root, for example `python app_code/find_negatives.py` or `python app_code/curated_negatives/run_lightonai_adaptive_pipeline.py`.

## Coding Style & Naming Conventions

Use Python 3.12, 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase environment variables. Ruff uses a 120-character line length, import sorting, bugbear checks, and pyupgrade rules. Prefer typed, small functions and helpers in `app_code/config.py` and `app_code/utils/` over ad hoc configuration or vector DB access.

## Testing Guidelines

Use pytest. Name files `test_*.py` and functions `test_*`. Keep tests deterministic with small synthetic data or temporary files; avoid large `data/` artifacts and external model downloads. For focused work, run `pytest tests/test_<module>.py`; before broader handoff, run `make check`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Fix memory calculation bug in batch tuning` and `Add nfcorpus_pl dataset and update config .env files`. Keep each commit focused on one concern.

Pull requests should describe the pipeline stage or module changed, list important config or `.env` variables, note generated artifacts intentionally updated, link related issues, and include validation output, usually `make check` or focused pytest.

## Security & Configuration Tips

Copy shareable examples such as `.env.example.nfcorpus-pl.two-stage` to `.env` for local runs. Do not commit private `.env` files, credentials, API tokens, or machine-specific paths. Prefer LanceDB for local no-service runs with `VECTOR_DB_BACKEND=lancedb` and `LANCEDB_PATH=./lancedb_data`.
