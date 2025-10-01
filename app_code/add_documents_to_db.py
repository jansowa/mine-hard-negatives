import argparse
import logging
import os

from models import get_dense_model, get_sparse_model
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from qdrant_client import QdrantClient, models
from decouple import config
from utils.vdb import get_qdrant_client
from tqdm import tqdm

from datasets import Dataset

# Suppress HTTP request logs from Qdrant client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def load_all_existing_ids(client: QdrantClient, collection_name: str, page_limit: int = 10000) -> set[str]:
    if not client.collection_exists(collection_name=collection_name):
        return set()

    try:
        total = client.count(collection_name=collection_name, exact=True).count
    except Exception:
        total = None

    existing: set[str] = set()
    offset = None
    pbar = tqdm(total=total, desc="Loading existing document_id from Qdrant", disable=(total is None))

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=page_limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not records:
            break
        for r in records:
            payload = r.payload or {}
            metadata = payload.get("metadata") or {}
            if "document_id" in metadata:
                existing.add(str(metadata["document_id"]))
        if total is not None:
            pbar.update(len(records))
        if next_offset is None:
            break
        offset = next_offset

    if total is not None:
        pbar.close()
    else:
        print(f"Loaded existing IDs: {len(existing)}")

    return existing


def filter_dataset_by_missing_ids(dataset_path: str, existing_ids: set[str], num_proc: int | None = None, batch_size: int = 10000) -> Dataset:
    ds = Dataset.from_parquet(dataset_path)

    if not existing_ids:
        return ds

    if num_proc is None:
        num_proc = max(os.cpu_count() or 1, 1)

    def _pred(batch):
        ids = batch["id"]
        return [str(x) not in existing_ids for x in ids]

    ds_filtered = ds.filter(_pred, batched=True, batch_size=batch_size, num_proc=num_proc)
    return ds_filtered


def process_file(dataset_path: str, dense_model_name: str, sparse_model_name: str,
                 batch_size: int, database_collection_name: str) -> None:
    client = get_qdrant_client()

    existing_ids = load_all_existing_ids(client, database_collection_name)
    ds_filtered = filter_dataset_by_missing_ids(dataset_path, existing_ids)
    total_in_parquet = len(Dataset.from_parquet(dataset_path))
    to_add_total = len(ds_filtered)

    print(f"Corpus items in parquet: {total_in_parquet}")
    print(f"Already present in Qdrant ({database_collection_name}): {len(existing_ids)}")
    print(f"Will add (after filtering): {to_add_total}")

    if to_add_total == 0:
        print("Nothing to add. Exiting.")
        return

    dense_embeddings = get_dense_model(dense_model_name, batch_size=batch_size)
    sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=batch_size)
    dense_dim_size = len(dense_embeddings.embed_query("text"))

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

    print(f"Number of points in collection before adding documents: "
          f"{get_points_number(client, database_collection_name)}")

    add_documents_from_dataset(ds_filtered, batch_size, vectorstore, total_hint=to_add_total)

    print(f"Number of points in collection after adding documents: "
          f"{get_points_number(client, database_collection_name)}")


def get_points_number(client: QdrantClient, collection_name: str) -> int:
    return client.count(
        collection_name=collection_name,
        exact=True,
    ).count


def add_documents_from_dataset(ds: Dataset, batch_size: int, vectorstore: QdrantVectorStore, total_hint: int | None = None) -> None:
    total = total_hint if total_hint is not None else len(ds)
    with tqdm(total=total, desc="Adding documents to vector database") as pbar:
        for batch in ds.iter(batch_size=batch_size):
            documents: list[Document] = []
            for content, doc_id in zip(batch['text'], batch['id']):
                if not content:
                    continue
                documents.append(Document(
                    page_content=content,
                    metadata={"document_id": doc_id}
                ))
            if documents:
                vectorstore.add_documents(documents=documents, wait=False)
                pbar.update(len(documents))


def create_collection_if_not_exists(client: QdrantClient, collection_name: str, dense_dim_size: int) -> None:
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=dense_dim_size, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the input parquet file.", default=config("CORPUS_PATH"))
    parser.add_argument("--dense_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, required=False, help="Name of sparse model to calculate embeddings", default=config("SPLADE_MODEL_NAME"))
    parser.add_argument("--batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, required=False, help="Name of database collection", default="all_documents")

    args = parser.parse_args()

    process_file(args.dataset_path, args.dense_model_name, args.sparse_model_name,
                 args.batch_size, args.database_collection_name)