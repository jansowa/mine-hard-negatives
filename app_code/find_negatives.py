from models import get_dense_model, get_sparse_model, get_reranker_model, rerank
import argparse
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import pandas as pd
from decouple import config
import pyarrow as pa
import pyarrow.parquet as pq
from utils.vdb import get_qdrant_client
import logging

# Suppress HTTP request logs from Qdrant client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def find_negatives(dense_model_name: str, sparse_model_name:str, embedding_batch_size: int, reranker_model_name: str, reranker_batch_size: int, processing_batch_size: int, collection_name: str, database_path: str, queries_path: str, relevant_path: str, output_path: str, top_k: int):
    dense_embeddings = get_dense_model(dense_model_name, batch_size=embedding_batch_size, prompt=config("DENSE_PROMPT"))
    print("Loaded dense embeddings")
    sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size)
    print("Loaded sparse embeddings")
    client = get_qdrant_client()
    print("Loaded database")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )
    tokenizer, reranker = get_reranker_model(model_name=reranker_model_name)
    print("Loaded reranker")
    queries_df = pd.read_parquet(queries_path)
    relative_df = pd.read_parquet(relevant_path)

    best_relative_df = relative_df.loc[relative_df.groupby('query_id')['positive_ranking'].idxmax()]
    positives_df = queries_df.merge(best_relative_df, left_on='id', right_on="query_id").drop(columns="id")

    negatives_schema = pa.schema([
        ('query_id', pa.int64()),
        ('document_id', pa.int64()),
        ('ranking', pa.float32()),
    ])
    n = len(positives_df)

    writer = pq.ParquetWriter(output_path, negatives_schema)
    buffer = []
    try:
        for i, (_, row) in enumerate(positives_df.iterrows(), 1):
            retrieved_docs = [
                document for document in retriever.get_relevant_documents(row['text'])
                if document.metadata['document_id'] != row['document_id']
            ]
            ranking = rerank(
                tokenizer, reranker, row['text'],
                [document.page_content for document in retrieved_docs],
                batch_size=reranker_batch_size
            )
            for document, rank in zip(retrieved_docs, ranking):
                buffer.append((row['query_id'], document.metadata['document_id'], rank))

            if i % 10 == 0:
                print(f"Found negatives for {i} queries.")
            if i % processing_batch_size == 0 or i == n:
                batch_df = pd.DataFrame(buffer, columns=["query_id", "document_id", "ranking"])
                batch_df = batch_df.astype({"query_id": "int64", "document_id": "int64", "ranking": "float32"})
                table = pa.Table.from_pandas(batch_df)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                print(f"Saved batch {i // processing_batch_size} ({i} of {n})")
                buffer = []
    except Exception as e:
        print(f"Error: {e}")
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--dense_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, required=False, help="Name of sparse model to calculate embeddings", default=config("SPLADE_MODEL_NAME"))
    parser.add_argument("--embedding_batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--reranker_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("RERANKER_NAME"))
    parser.add_argument("--reranker_batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("RERANKER_BATCH_SIZE", cast=int))
    parser.add_argument("--processing_batch_size", type=int, required=False, help="Number of questions in single iteration before save", default=config("PROCESSING_CHUNK_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, required=False, help="Name of database collection", default="all_documents")
    parser.add_argument("--database_path", type=str, required=False, help="Path to the output JSONL file (optional).",
                        default="qdrant_db")
    parser.add_argument("--queries_path", type=str, required=False, help="Path to the queries parquet file.", default=config("QUERIES_PATH"))
    parser.add_argument("--relevant_path", type=str, required=False, help="Path to the relevancy parquet file.", default=config("RELEVANT_WITH_SCORE_PATH"))
    parser.add_argument("--output_path", type=str, required=False, help="Path to the output parquet file.", default=config("NEGATIVES_PATH"))
    parser.add_argument("--top_k", type=int, default=config("TOP_K", cast=int), required=False, help="Number of documents to retrieve")
    args = parser.parse_args()
    find_negatives(args.dense_model_name, args.sparse_model_name, args.embedding_batch_size, args.reranker_model_name, args.reranker_batch_size, args.processing_batch_size,
                   args.database_collection_name, args.database_path, args.queries_path, args.relevant_path, args.output_path, args.top_k)
