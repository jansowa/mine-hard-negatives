import argparse
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset
from decouple import config
from langchain_core.documents import Document
from tqdm import tqdm

from batch_tuning import (
    benchmark_embedding_batch_size as benchmark_dense_batch_size,
)
from batch_tuning import (
    embed_dense_documents,
    parse_batch_size_candidates,
    set_dense_embedding_batch_size,
)
from config import get_vector_db_backend
from models import get_dense_model, get_sparse_model
from utils.vector_db import VectorBackend, create_vector_backend


def filter_dataset_by_missing_ids_ds(
    ds: Dataset,
    existing_ids: set[str],
    num_proc: int | None = None,
    batch_size: int = 10000,
) -> Dataset:
    if not existing_ids:
        return ds

    if num_proc is None:
        num_proc = 1

    def _pred(batch):
        return [str(x) not in existing_ids for x in batch["id"]]

    if num_proc <= 1:
        return ds.filter(_pred, batched=True, batch_size=batch_size)

    return ds.filter(_pred, batched=True, batch_size=batch_size, num_proc=num_proc)


def add_documents_from_dataset(
    ds: Dataset,
    batch_size: int,
    backend: VectorBackend,
    total_hint: int | None = None,
    db_write_batch_size: int | None = None,
) -> None:
    total = total_hint if total_hint is not None else len(ds)
    write_batch_size = db_write_batch_size or batch_size
    if write_batch_size <= 0:
        raise ValueError("db_write_batch_size must be greater than 0")

    with tqdm(total=total, desc="Adding documents to vector database") as pbar:
        for batch in ds.iter(batch_size=write_batch_size):
            documents: list[Document] = []
            for content, doc_id in zip(batch["text"], batch["id"]):
                if not content:
                    continue
                documents.append(Document(page_content=content, metadata={"document_id": str(doc_id)}))

            if documents:
                backend.upsert_documents(documents)
                pbar.update(len(documents))


def collect_text_sample(ds: Dataset, sample_size: int) -> list[str]:
    if sample_size <= 0:
        return []

    sample: list[str] = []
    batch_size = min(max(sample_size, 1), 10_000)
    for batch in ds.iter(batch_size=batch_size):
        for text in batch["text"]:
            if text:
                sample.append(text)
                if len(sample) >= sample_size:
                    return sample
    return sample


def write_lancedb_embedding_batch(
    backend: VectorBackend,
    document_ids: list[str],
    texts: list[str],
    vectors,
) -> int:
    backend.upsert_embeddings(document_ids=document_ids, texts=texts, vectors=vectors)
    return len(texts)


def add_documents_to_lancedb_from_dataset(
    ds: Dataset,
    dense_embeddings,
    backend: VectorBackend,
    total_hint: int | None = None,
    db_write_batch_size: int | None = None,
    async_write: bool = True,
) -> None:
    total = total_hint if total_hint is not None else len(ds)
    write_batch_size = db_write_batch_size or config("LANCEDB_DB_WRITE_BATCH_SIZE", cast=int, default=4096)
    if write_batch_size <= 0:
        raise ValueError("db_write_batch_size must be greater than 0")

    executor = ThreadPoolExecutor(max_workers=1) if async_write else None
    pending_write = None

    try:
        with tqdm(total=total, desc="Adding documents to LanceDB") as pbar:
            for batch in ds.iter(batch_size=write_batch_size):
                document_ids: list[str] = []
                texts: list[str] = []
                for content, doc_id in zip(batch["text"], batch["id"]):
                    if not content:
                        continue
                    document_ids.append(str(doc_id))
                    texts.append(content)

                if not texts:
                    continue

                vectors = embed_dense_documents(dense_embeddings, texts)

                if pending_write is not None:
                    pbar.update(pending_write.result())
                    pending_write = None

                if executor is None:
                    backend.upsert_embeddings(document_ids=document_ids, texts=texts, vectors=vectors)
                    pbar.update(len(texts))
                else:
                    pending_write = executor.submit(
                        write_lancedb_embedding_batch,
                        backend,
                        document_ids,
                        texts,
                        vectors,
                    )

            if pending_write is not None:
                pbar.update(pending_write.result())
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def process_file(
    dataset_path: str,
    dense_model_name: str,
    sparse_model_name: str,
    batch_size: int | None,
    db_write_batch_size: int,
    database_collection_name: str,
    skip: int = 0,
    offset: int | None = None,
    resume: bool = True,
    compact_existing: str = "none",
    auto_batch_size_candidates: str | None = None,
    auto_batch_size_min: int = 8,
    auto_batch_size_max: int = 256,
    auto_batch_size_sample_size: int = 512,
    lancedb_async_write: bool = True,
) -> None:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")
    if db_write_batch_size <= 0:
        raise ValueError("--db_write_batch_size must be greater than 0")
    if auto_batch_size_min <= 0 or auto_batch_size_max <= 0:
        raise ValueError("--auto_batch_size_min and --auto_batch_size_max must be greater than 0")
    if auto_batch_size_min > auto_batch_size_max:
        raise ValueError("--auto_batch_size_min cannot be greater than --auto_batch_size_max")

    ds_all = Dataset.from_parquet(dataset_path)
    total_in_parquet = len(ds_all)

    if skip < 0:
        skip = 0
    if skip >= total_in_parquet:
        print(f"Corpus items in parquet: {total_in_parquet}")
        print(f"Requested window starts beyond dataset length (skip={skip}). Nothing to do. Exiting.")
        return

    end_idx = total_in_parquet if offset is None else min(total_in_parquet, skip + max(offset, 0))
    ds_window = ds_all.select(range(skip, end_idx))

    backend_type = get_vector_db_backend()
    effective_batch_size = batch_size

    if backend_type == "qdrant":
        effective_batch_size = batch_size or config("EMBEDDER_BATCH_SIZE", cast=int)
        dense_embeddings = get_dense_model(dense_model_name, batch_size=effective_batch_size)
        sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=effective_batch_size)
        dense_dim_size = len(dense_embeddings.embed_query("text"))
    else:
        dense_embeddings = None
        sparse_embeddings = None
        dense_dim_size = None

    backend = create_vector_backend(
        collection_name=database_collection_name,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
        dense_dim_size=dense_dim_size,
    )

    if compact_existing != "none":
        print(f"Preparing existing backend table with compact_existing={compact_existing!r}...")
        backend.compact_existing(compact_existing)
        backend.print_storage_stats()

    if resume:
        print("Scanning existing document_id values for resume...")
        existing_ids = backend.existing_document_ids()
        ds_filtered = filter_dataset_by_missing_ids_ds(ds_window, existing_ids)
    else:
        existing_ids = set()
        ds_filtered = ds_window

    print(f"Corpus items in parquet: {total_in_parquet}")
    print(f"Selected window (skip={skip}, offset={offset}): {len(ds_window)}")
    print(f"Resume enabled: {resume}")
    print(f"Already present in backend ({database_collection_name}): {len(existing_ids)}")
    print(f"Will add from selected window (after filtering): {len(ds_filtered)}")
    effective_db_write_batch_size = db_write_batch_size if backend_type == "lancedb" else effective_batch_size

    if len(ds_filtered) == 0:
        print("Optimizing backend where supported...")
        backend.optimize()
        print("Ensuring vector/text indexes where supported...")
        backend.ensure_indexes()
        backend.print_storage_stats()
        print("Index check completed.")
        print("Nothing to add. Exiting.")
        return

    if backend_type == "lancedb":
        initial_batch_size = batch_size or auto_batch_size_min
        dense_embeddings = get_dense_model(dense_model_name, batch_size=initial_batch_size)
        if batch_size is None:
            candidates = parse_batch_size_candidates(
                auto_batch_size_candidates,
                minimum=auto_batch_size_min,
                maximum=auto_batch_size_max,
            )
            sample_texts = collect_text_sample(ds_filtered, auto_batch_size_sample_size)
            effective_batch_size = benchmark_dense_batch_size(dense_embeddings, sample_texts, candidates)
        else:
            effective_batch_size = batch_size
            set_dense_embedding_batch_size(dense_embeddings, effective_batch_size)
        backend.set_dense_embeddings(dense_embeddings)

    if effective_batch_size is None:
        raise ValueError("Embedding batch size could not be determined")

    print(f"Embedding batch size: {effective_batch_size}")
    print(f"DB write batch size: {effective_db_write_batch_size}")
    if backend_type == "lancedb":
        print(f"LanceDB async write: {lancedb_async_write}")

    print(f"Number of points before adding documents: {backend.count()}")
    if backend_type == "lancedb":
        add_documents_to_lancedb_from_dataset(
            ds_filtered,
            dense_embeddings,
            backend,
            total_hint=len(ds_filtered),
            db_write_batch_size=effective_db_write_batch_size,
            async_write=lancedb_async_write,
        )
    else:
        add_documents_from_dataset(
            ds_filtered,
            effective_batch_size,
            backend,
            total_hint=len(ds_filtered),
            db_write_batch_size=effective_db_write_batch_size,
        )
    print(f"Number of points after adding documents: {backend.count()}")
    print("Optimizing backend where supported...")
    backend.optimize()
    print("Ensuring vector/text indexes where supported...")
    backend.ensure_indexes()
    backend.print_storage_stats()
    print("Index check completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add documents to vector DB with optional skip/offset windowing.")
    parser.add_argument("--dataset_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--dense_model_name", type=str, default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, default=config("SPLADE_MODEL_NAME"))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Embedding microbatch size. If omitted for LanceDB, a short benchmark selects it automatically.",
    )
    parser.add_argument(
        "--auto_batch_size_candidates",
        type=str,
        default=config("ADD_DOCUMENTS_AUTO_BATCH_SIZE_CANDIDATES", default=None),
    )
    parser.add_argument(
        "--auto_batch_size_min", type=int, default=config("ADD_DOCUMENTS_AUTO_BATCH_SIZE_MIN", cast=int, default=8)
    )
    parser.add_argument(
        "--auto_batch_size_max", type=int, default=config("ADD_DOCUMENTS_AUTO_BATCH_SIZE_MAX", cast=int, default=256)
    )
    parser.add_argument(
        "--auto_batch_size_sample_size",
        type=int,
        default=config("ADD_DOCUMENTS_AUTO_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=512),
    )
    parser.add_argument(
        "--db_write_batch_size",
        type=int,
        default=config("LANCEDB_DB_WRITE_BATCH_SIZE", cast=int, default=4096),
    )
    parser.add_argument(
        "--database_collection_name", type=str, default=config("DATABASE_COLLECTION_NAME", default="all_documents")
    )
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=config("ADD_DOCUMENTS_RESUME", cast=bool, default=True))
    parser.add_argument("--compact_existing", choices=["none", "optimize", "rebuild"], default="none")
    parser.add_argument(
        "--lancedb_async_write",
        dest="lancedb_async_write",
        action="store_true",
        help="Overlap embedding of the next large chunk with LanceDB writing of the previous chunk.",
    )
    parser.add_argument("--no_lancedb_async_write", dest="lancedb_async_write", action="store_false")
    parser.set_defaults(lancedb_async_write=config("LANCEDB_ASYNC_WRITE", cast=bool, default=True))
    args = parser.parse_args()

    process_file(
        dataset_path=args.dataset_path,
        dense_model_name=args.dense_model_name,
        sparse_model_name=args.sparse_model_name,
        batch_size=args.batch_size,
        db_write_batch_size=args.db_write_batch_size,
        database_collection_name=args.database_collection_name,
        skip=args.skip,
        offset=args.offset,
        resume=args.resume,
        compact_existing=args.compact_existing,
        auto_batch_size_candidates=args.auto_batch_size_candidates,
        auto_batch_size_min=args.auto_batch_size_min,
        auto_batch_size_max=args.auto_batch_size_max,
        auto_batch_size_sample_size=args.auto_batch_size_sample_size,
        lancedb_async_write=args.lancedb_async_write,
    )
