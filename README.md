# Project to mine hard negatives and set scores for knowledge distillation

### How to run?
1. Add data to `data/input.jsonl` - use schema of sample file provided in the repository.
2. Create `.env` file - you can use `.env.example.base` with models based on "roberta base" or `.env.example.large` with models based on "roberta large".
3. Run:
```shell
python to_huggingface_dataset.py
```
This script will generate the files `corpus.parquet`, `queries.parquet`, and `relevant.parquet` in the `data` folder based on the `data/sample_input.jsonl` file.
4. Add documents to the database (default is folder `qdrant_db` and collection `all_documents`):
```shell
python add_documents_to_db.py
```
5. Add scores to the "positives" in the `relevant_with_score.parquet` file:
```shell
python add_positives_ranks.py
```
6. Add computed negatives:
```shell
python find_negatives.py
```

7. Create a FlagEmbedding-style JSONL with a training-ready dataset:
```shell
python create_flag_embedding_jsonl.py
```