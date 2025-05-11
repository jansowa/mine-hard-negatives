# Project to mine hard negatives and set scores for knowledge distillation

### How to run?
1. Run:
```shell
python to_huggingface_dataset.py
```
This script will generate the files `corpus.parquet`, `queries.parquet`, and `relevant.parquet` in the `data` folder based on the `data/sample_input.jsonl` file.
2. Add documents to the database (default is folder `qdrant_db` and collection `all_documents`):
```shell
python add_documents_to_db.py --dataset_path "data/corpus.parquet"
```
3. Add scores to the "positives" in the `relevant_with_score.parquet` file:
```shell
python add_positives_ranks.py
```
4. Add computed negatives:
```shell
python find_negatives.py
```

5. Create a FlagEmbedding-style JSONL with a training-ready dataset:
```shell
create_flag_embedding_jsonl.py
```