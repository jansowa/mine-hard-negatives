from models import get_dense_model, get_sparse_model, get_reranker_model, rerank
import argparse
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import pandas as pd
from decouple import config


def find_negatives(dense_model_name: str, sparse_model_name:str, embedding_batch_size: int, reranker_model_name: str, reranker_batch_size: int, collection_name: str, database_path: str, queries_path: str, relevant_path: str, output_path: str, top_k: int):
    dense_embeddings = get_dense_model(dense_model_name, batch_size=embedding_batch_size, prompt=config("DENSE_PROMPT"))
    print("Loaded dense embeddings")
    sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size)
    print("Loaded sparse embeddings")
    client = QdrantClient(path=database_path)
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

    zipped = []
    for index, row in positives_df.iterrows():
        retrieved_docs = [document for document in retriever.get_relevant_documents(row['text']) if
                          document.metadata['document_id'] != row['document_id']]
        ranking = rerank(tokenizer, reranker, row['text'], [document.page_content for document in retrieved_docs],
                         batch_size=reranker_batch_size)
        zipped = zipped + [(row['query_id'], document_id, ranking) for document_id, ranking in
                           zip([document.metadata['document_id'] for document in retrieved_docs], ranking)]
    negatives = pd.DataFrame(zipped, columns=["query_id", "document_id", "ranking"])
    negatives.to_parquet(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--dense_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, required=False, help="Name of sparse model to calculate embeddings", default="sdadas/polish-splade")
    parser.add_argument("--embedding_batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--reranker_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("RERANKER_NAME"))
    parser.add_argument("--reranker_batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("RERANKER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, required=False, help="Name of database collection", default="all_documents")
    parser.add_argument("--database_path", type=str, required=False, help="Path to the output JSONL file (optional).",
                        default="qdrant_db")
    parser.add_argument("--queries_path", type=str, required=False, help="Path to the queries parquet file.", default="data/queries.parquet")
    parser.add_argument("--relevant_path", type=str, required=False, help="Path to the relevancy parquet file.", default="data/relevant_with_score.parquet")
    parser.add_argument("--output_path", type=str, required=False, help="Path to the output parquet file.", default="data/negatives.parquet")
    parser.add_argument("--top_k", type=int, default=config("TOP_K", cast=int), required=False, help="Number of documents to retrieve")
    args = parser.parse_args()
    find_negatives(args.dense_model_name, args.sparse_model_name, args.embedding_batch_size, args.reranker_model_name, args.reranker_batch_size, args.database_collection_name, args.database_path, args.queries_path, args.relevant_path,
                   args.output_path, args.top_k)
