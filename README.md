# Hard negatives mining pipeline

Projekt służy do budowania datasetu treningowego (query, positive, hard negatives) dla embedderów i rerankerów.

## Tryby uruchomienia

### 1) venv + pip (zalecane na HPC, bez Dockera)

1. Utwórz i aktywuj środowisko:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. Przygotuj `.env` (na bazie `.env.example.base` lub `.env.example.large`).
3. Ustaw backend bazy wektorowej:
   - LanceDB (domyślny, bez Dockera):
     ```bash
     export VECTOR_DB_BACKEND=lancedb
     export LANCEDB_PATH=./lancedb_data
     ```
   - Qdrant (opcjonalnie):
     ```bash
     export VECTOR_DB_BACKEND=qdrant
     export QDRANT_URL=http://localhost:6333
     ```

### 2) Docker Compose (opcjonalny wygodny local setup)

```bash
docker compose up -d
```

To uruchamia kontener `executable` oraz (opcjonalny) serwis `vdb` oparty o Qdrant.

## Pipeline

1. Przygotuj dane (`corpus.parquet`, `queries.parquet`, `relevant.parquet`):
   ```bash
   python app_code/to_huggingface_dataset.py
   # albo
   python app_code/msmarco_to_huggingface_dataset.py
   # albo
   python app_code/clips_mqa_to_huggingface_dataset.py
   ```
2. Załaduj dokumenty do bazy wektorowej:
   ```bash
   python app_code/add_documents_to_db.py
   ```
3. Dodaj score dla pozytywów:
   ```bash
   python app_code/add_positives_ranks.py
   ```
4. Wykop negatywy:
   ```bash
   python app_code/find_negatives.py
   ```
5. Zbuduj JSONL pod FlagEmbedding:
   ```bash
   python app_code/create_flag_embedding_jsonl.py
   ```

## Uwaga o backendach

- `VECTOR_DB_BACKEND=lancedb` używa dense retrieval + BM25/hybrid LanceDB.
- `VECTOR_DB_BACKEND=qdrant` zachowuje dotychczasowy przepływ dense+sparse (SPLADE).
