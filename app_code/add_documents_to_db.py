import argparse
import os

from datasets import Dataset
from decouple import config
from langchain_core.documents import Document
from tqdm import tqdm

from config import get_vector_db_backend
from models import get_dense_model, get_sparse_model
from utils.vector_db import create_vector_backend, VectorBackend


def filter_dataset_by_missing_ids_ds(
    ds: Dataset,
    existing_ids: set[str],
    num_proc: int | None = None,
    batch_size: int = 10000,
) -> Dataset:
    if not existing_ids:
        return ds

    if num_proc is None:
        num_proc = max(os.cpu_count() or 1, 1)

    def _pred(batch):
        return [str(x) not in existing_ids for x in batch["id"]]

    return ds.filter(_pred, batched=True, batch_size=batch_size, num_proc=num_proc)


def add_documents_from_dataset(
    ds: Dataset,
    batch_size: int,
    backend: VectorBackend,
    total_hint: int | None = None,
) -> None:
    total = total_hint if total_hint is not None else len(ds)
    with tqdm(total=total, desc="Adding documents to vector database") as pbar:
        for batch in ds.iter(batch_size=batch_size):
            documents: list[Document] = []
            for content, doc_id in zip(batch["text"], batch["id"]):
                if not content:
                    continue
                documents.append(Document(page_content=content, metadata={"document_id": str(doc_id)}))

            if documents:
                backend.upsert_documents(documents)
                pbar.update(len(documents))


def process_file(
    dataset_path: str,
    dense_model_name: str,
    sparse_model_name: str,
    batch_size: int,
    database_collection_name: str,
    skip: int = 0,
    offset: int | None = None,
) -> None:
    ds_all = Dataset.from_parquet(dataset_path)
    total_in_parquet = len(ds_all)

    if skip < 0:
        skip = 0
    if skip >= total_in_parquet:
        print(f"Corpus items in parquet: {total_in_parquet}")
        print(f"Requested window starts beyond dataset length (skip={skip}). Nothing to do. Exiting.")
        return

    end_idx = total_in_parquet if offset is None else min(total_in_parquet, skip + max(offset, 0))
    ds_window = ds_all.select(list(range(skip, end_idx)))

    dense_embeddings = get_dense_model(dense_model_name, batch_size=batch_size)
    backend_type = get_vector_db_backend()

    sparse_embeddings = None
    dense_dim_size = None
    if backend_type == "qdrant":
        sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=batch_size)
        dense_dim_size = len(dense_embeddings.embed_query("text"))

    backend = create_vector_backend(
        collection_name=database_collection_name,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
        dense_dim_size=dense_dim_size,
    )

    existing_ids = backend.existing_document_ids()
    ds_filtered = filter_dataset_by_missing_ids_ds(ds_window, existing_ids)

    print(f"Corpus items in parquet: {total_in_parquet}")
    print(f"Selected window (skip={skip}, offset={offset}): {len(ds_window)}")
    print(f"Already present in backend ({database_collection_name}): {len(existing_ids)}")
    print(f"Will add from selected window (after filtering): {len(ds_filtered)}")

    if len(ds_filtered) == 0:
        print("Nothing to add. Exiting.")
        return

    print(f"Number of points before adding documents: {backend.count()}")
    add_documents_from_dataset(ds_filtered, batch_size, backend, total_hint=len(ds_filtered))
    print(f"Number of points after adding documents: {backend.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add documents to vector DB with optional skip/offset windowing.")
    parser.add_argument("--dataset_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--dense_model_name", type=str, default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, default=config("SPLADE_MODEL_NAME"))
    parser.add_argument("--batch_size", type=int, default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, default="all_documents")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--offset", type=int, default=None)
    args = parser.parse_args()

    process_file(
        dataset_path=args.dataset_path,
        dense_model_name=args.dense_model_name,
        sparse_model_name=args.sparse_model_name,
        batch_size=args.batch_size,
        database_collection_name=args.database_collection_name,
        skip=args.skip,
        offset=args.offset,
    )
