import argparse
from models import get_dense_model, get_sparse_model
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from qdrant_client import QdrantClient, models
from decouple import config

from datasets import Dataset

def read_parquet_batches(file_path: str, batch_size: int):
    ds = Dataset.from_parquet(file_path)
    for batch in ds.iter(batch_size=batch_size):
        yield batch

def process_file(dataset_path: str, database_path: str, dense_model_name: str, sparse_model_name:str,
                 batch_size: int, database_collection_name: str):
    dense_embeddings = get_dense_model(dense_model_name, batch_size=batch_size)
    sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=batch_size)
    dense_dim_size = len(dense_embeddings.embed_query("text"))

    client = QdrantClient(path=database_path)
    create_collection_if_not_exists(client, database_collection_name, dense_dim_size)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=database_collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    print(f"Number of points in collection before adding documents: {get_points_number(client, database_collection_name)}")
    add_documents(dataset_path, batch_size, vectorstore)
    print(f"Number of points in collection after adding documents: {get_points_number(client, database_collection_name)}")


def get_points_number(client: QdrantClient, collection_name: str) -> int:
    return client.count(
        collection_name=collection_name,
        exact=True,
    ).count



def add_documents(input_file: str, lines_number_batch: int, vectorstore: QdrantVectorStore):

    for batch in read_parquet_batches(input_file, lines_number_batch):
        documents: list[Document] = []
        for content, doc_id in zip(batch['text'], batch['id']):
            if not content:
                continue
            documents.append(Document(
                page_content=content,
                metadata={"document_id": doc_id}
            ))
        if documents:
            vectorstore.add_documents(documents=documents)


def create_collection_if_not_exists(client, database_collection_name, dense_dim_size):
    if not client.collection_exists(collection_name=database_collection_name):
        client.create_collection(
            collection_name=database_collection_name,
            vectors_config={"dense": VectorParams(size=dense_dim_size, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the input parquet file.", default=config("CORPUS_PATH"))
    parser.add_argument("--database_path", type=str, required=False, help="Path to the qdrant output database", default="qdrant_db")
    parser.add_argument("--dense_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, required=False, help="Name of sparse model to calculate embeddings", default="sdadas/polish-splade")
    parser.add_argument("--batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, required=False, help="Name of database collection", default="all_documents")

    args = parser.parse_args()

    process_file(args.dataset_path, args.database_path, args.dense_model_name, args.sparse_model_name,
                 args.batch_size, args.database_collection_name)