# Project to mine hard negatives and set scores for knowledge distillation

### How to run?
1. Create `.env` file - you can use `.env.example.base` with models based on "roberta base" or `.env.example.large` with models based on "roberta large".
2. Prepare your data in proper form. You can run:
   - `python to_huggingface_dataset.py` to transform `data/input.jsonl` into parquets
   - `python msmarco_to_huggingface_dataset.py` to transform MS Marco into parquets

   This script will generate the files `corpus.parquet`, `queries.parquet`, and `relevant.parquet` in the `data` folder.  
   Expected structure:  
   - **corpus.parquet**  
     - `id` — document id (int32)  
     - `text` — document text (string)  
   - **queries.parquet**  
     - `id` — query id (int32)  
     - `text` — query text (string)  
   - **relevant.parquet**  
     - `query_id` (int32)  
     - `document_id` (int32)  
3. Add documents to the database (default is folder `qdrant_db` and collection `all_documents`):
   ```shell
   python add_documents_to_db.py
   ```
4. Add scores to the "positives" in the `relevant_with_score.parquet` file:
   ```shell
   python add_positives_ranks.py
   ```
5. Add computed negatives:
   ```shell
   python find_negatives.py
   ```
6. Create a FlagEmbedding-style JSONL with a training-ready dataset:
   ```shell
   python create_flag_embedding_jsonl.py
   ```