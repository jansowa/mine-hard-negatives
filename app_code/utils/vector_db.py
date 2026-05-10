from __future__ import annotations

import os
import random
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from decouple import config
from langchain_core.documents import Document

from config import get_lancedb_path, get_qdrant_url, get_vector_db_backend


class VectorBackend(ABC):
    @abstractmethod
    def upsert_documents(self, documents: list[Document]) -> None: ...

    def upsert_embeddings(self, document_ids: list[str], texts: list[str], vectors) -> None:
        raise NotImplementedError("This backend does not support precomputed dense embedding upserts")

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def search(self, query_text: str, k: int, offset: int = 0) -> list[Document]: ...

    def search_many_offsets(self, query_text: str, k: int, offsets: list[int]) -> dict[int, list[Document]]:
        return {offset: self.search(query_text=query_text, k=k, offset=offset) for offset in offsets}

    @abstractmethod
    def random_sample(self, k: int) -> list[Document]: ...

    @abstractmethod
    def existing_document_ids(self) -> set[str]: ...

    def ensure_indexes(self) -> None:
        return None

    def optimize(self) -> None:
        return None

    def compact_existing(self, mode: str = "none") -> None:
        return None

    def print_storage_stats(self) -> None:
        return None

    def set_dense_embeddings(self, dense_embeddings) -> None:
        return None


class QdrantBackend(VectorBackend):
    def __init__(
        self,
        collection_name: str,
        dense_embeddings,
        sparse_embeddings,
        dense_dim_size: int,
        qdrant_url: str | None = None,
    ) -> None:
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        from qdrant_client import QdrantClient, models
        from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url or get_qdrant_url())

        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=dense_dim_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    def upsert_documents(self, documents: list[Document]) -> None:
        self.vector_store.add_documents(documents=documents, wait=False)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def search(self, query_text: str, k: int, offset: int = 0) -> list[Document]:
        docs = self.vector_store.similarity_search(query=query_text, k=k, offset=offset)
        for local_rank, doc in enumerate(docs):
            metadata = dict(doc.metadata or {})
            metadata.setdefault("retrieval_rank", offset + local_rank)
            metadata.setdefault("retrieval_offset", offset)
            metadata.setdefault("retrieval_score", None)
            metadata.setdefault("retrieval_source", "qdrant")
            doc.metadata = metadata
        return docs

    def random_sample(self, k: int) -> list[Document]:
        out = []
        next_page = None

        for _ in range(10):
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=min(k - len(out), 256),
                offset=next_page,
            )
            random.shuffle(points)
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                page_content = payload.get("page_content") or payload.get("text") or ""
                if "document_id" in metadata:
                    metadata = dict(metadata)
                    metadata.setdefault("retrieval_rank", len(out))
                    metadata.setdefault("retrieval_offset", None)
                    metadata.setdefault("retrieval_score", None)
                    metadata.setdefault("retrieval_source", "random")
                    out.append(Document(page_content=page_content, metadata=metadata))
                if len(out) >= k:
                    break
            if len(out) >= k or next_page is None:
                break

        return out[:k]

    def existing_document_ids(self) -> set[str]:
        existing: set[str] = set()
        offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10_000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            if not records:
                break

            for record in records:
                payload = record.payload or {}
                metadata = payload.get("metadata") or {}
                if "document_id" in metadata:
                    existing.add(str(metadata["document_id"]))

            if next_offset is None:
                break
            offset = next_offset

        return existing


class LanceDBBackend(VectorBackend):
    HYBRID_SELECT_COLUMNS = ["document_id", "text", "_distance", "_score"]
    VECTOR_SELECT_COLUMNS = ["document_id", "text", "_distance"]

    def __init__(self, collection_name: str, dense_embeddings, lancedb_path: str | None = None) -> None:
        import lancedb

        self.collection_name = collection_name
        self.dense_embeddings = dense_embeddings
        self.lancedb_path = lancedb_path or get_lancedb_path()
        self.nprobes = config("LANCEDB_NPROBES", cast=int, default=20)
        self.refine_factor = config("LANCEDB_REFINE_FACTOR", cast=int, default=2)
        self.target_partition_size = config("LANCEDB_TARGET_PARTITION_SIZE", cast=int, default=4096)
        self.query_vector_cache_size = config("LANCEDB_QUERY_VECTOR_CACHE_SIZE", cast=int, default=10_000)
        self._query_vector_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_vector_cache_lock = threading.Lock()
        self._random_sample_rows_cache = None
        self._random_sample_rows_lock = threading.Lock()
        self.db = lancedb.connect(self.lancedb_path)
        self.table = self._load_table_if_exists(collection_name)

    def _load_table_if_exists(self, collection_name: str):
        if hasattr(self.db, "list_tables"):
            raw_table_names = self.db.list_tables()
        else:
            raw_table_names = self.db.table_names()

        table_names = set()
        if hasattr(raw_table_names, "tables"):
            table_names.update(str(name) for name in raw_table_names.tables)

        for item in raw_table_names:
            if isinstance(item, str):
                table_names.add(item)
            elif isinstance(item, (list, tuple)) and item:
                key = str(item[0])
                value = item[1] if len(item) > 1 else None
                if key == "tables" and isinstance(value, list):
                    table_names.update(str(name) for name in value)
                elif key not in {"tables", "page_token"}:
                    table_names.add(key)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("table_name")
                if name:
                    table_names.add(str(name))
            elif hasattr(item, "name"):
                table_names.add(str(item.name))

        if collection_name in table_names:
            return self.db.open_table(collection_name)

        try:
            return self.db.open_table(collection_name)
        except Exception:
            pass

        return None

    def set_dense_embeddings(self, dense_embeddings) -> None:
        self.dense_embeddings = dense_embeddings
        self._clear_query_vector_cache()

    def upsert_embeddings(self, document_ids: list[str], texts: list[str], vectors) -> None:
        if not document_ids:
            return
        if len(document_ids) != len(texts):
            raise ValueError(f"document_ids/texts length mismatch: {len(document_ids)} != {len(texts)}")

        vector_array = self._vectors_to_numpy(vectors)
        if vector_array.shape[0] != len(texts):
            raise ValueError(f"texts/vectors length mismatch: {len(texts)} != {vector_array.shape[0]}")
        if vector_array.shape[1] == 0:
            raise ValueError("Dense embedding returned empty vector")

        data = self._arrow_rows(document_ids=document_ids, texts=texts, vectors=vector_array)
        self._add_lancedb_rows(data)

    @staticmethod
    def _normalise_vectors(vectors: Iterable[Iterable[float]]) -> list[list[float]]:
        values = [list(vector) for vector in vectors]
        if not values:
            return values

        dim = len(values[0])
        if dim == 0:
            raise ValueError("Dense embedding returned empty vector")

        for vector in values:
            if len(vector) != dim:
                raise ValueError(f"Inconsistent dense embedding dimensions in one batch: {dim} and {len(vector)}")

        return values

    @staticmethod
    def _vectors_to_numpy(vectors):
        import numpy as np

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Dense embeddings must be a 2D array, got shape {array.shape}")
        return array

    @staticmethod
    def _arrow_rows(document_ids: list[str], texts: list[str], vectors):
        import pyarrow as pa

        dim = vectors.shape[1]
        vector_values = pa.array(vectors.reshape(-1), type=pa.float32())
        vector_column = pa.FixedSizeListArray.from_arrays(vector_values, dim)
        return pa.table(
            {
                "document_id": pa.array([str(doc_id) for doc_id in document_ids], type=pa.string()),
                "text": pa.array(texts, type=pa.string()),
                "vector": vector_column,
            }
        )

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        if self.dense_embeddings is None:
            raise ValueError("Dense embeddings must be configured before adding LanceDB documents")

        texts = [d.page_content for d in documents]
        vectors = self._normalise_vectors(self.dense_embeddings.embed_documents(texts))
        document_ids = [str(doc.metadata.get("document_id")) for doc in documents]
        self.upsert_embeddings(document_ids=document_ids, texts=texts, vectors=vectors)

    def _add_lancedb_rows(self, rows) -> None:
        self._clear_random_sample_cache()
        self._clear_query_vector_cache()
        if self.table is None:
            table_dir = self._table_dir(self.collection_name)
            if table_dir.exists():
                self._reconnect()
            if self.table is None and table_dir.exists():
                raise RuntimeError(
                    f"LanceDB table directory exists but table {self.collection_name!r} could not be opened. "
                    "Refusing to overwrite existing data."
                )
            self.table = self.db.create_table(self.collection_name, data=rows, mode="overwrite")
        else:
            self.table.add(rows)

    def count(self) -> int:
        if self.table is None:
            return 0
        return self.table.count_rows()

    def search(self, query_text: str, k: int, offset: int = 0) -> list[Document]:
        if self.table is None:
            return []
        if self.dense_embeddings is None:
            raise ValueError("Dense embeddings must be configured before LanceDB search")

        vector = self._embed_query_cached(query_text)
        limit = max(k + offset, k)

        # Prefer hybrid query: dense + BM25 FTS.
        retrieval_source = "hybrid"
        try:
            hits = (
                self._tune_query(self.table.search(query_type="hybrid").vector(vector).text(query_text))
                .select(self.HYBRID_SELECT_COLUMNS)
                .limit(limit)
                .to_list()
            )
        except Exception:
            retrieval_source = "vector"
            hits = self._tune_query(self.table.search(vector)).select(self.VECTOR_SELECT_COLUMNS).limit(limit).to_list()

        docs = []
        for local_rank, hit in enumerate(hits[offset : offset + k]):
            doc_id = str(hit.get("document_id", ""))
            docs.append(
                Document(
                    page_content=hit.get("text", ""),
                    metadata={
                        "document_id": doc_id,
                        "retrieval_rank": offset + local_rank,
                        "retrieval_offset": offset,
                        "retrieval_score": hit.get("_score", hit.get("_distance")),
                        "retrieval_source": retrieval_source,
                    },
                )
            )

        return docs[:k]

    def search_many_offsets(self, query_text: str, k: int, offsets: list[int]) -> dict[int, list[Document]]:
        if self.table is None:
            return {offset: [] for offset in offsets}
        if self.dense_embeddings is None:
            raise ValueError("Dense embeddings must be configured before LanceDB search")

        offsets = sorted(set(offsets))
        if not offsets:
            return {}

        vector = self._embed_query_cached(query_text)
        limit = max(offsets) + k

        # Prefer hybrid query: dense + BM25 FTS.
        retrieval_source = "hybrid"
        try:
            hits = (
                self._tune_query(self.table.search(query_type="hybrid").vector(vector).text(query_text))
                .select(self.HYBRID_SELECT_COLUMNS)
                .limit(limit)
                .to_list()
            )
        except Exception:
            retrieval_source = "vector"
            hits = self._tune_query(self.table.search(vector)).select(self.VECTOR_SELECT_COLUMNS).limit(limit).to_list()

        out: dict[int, list[Document]] = {}
        for offset in offsets:
            docs = []
            for local_rank, hit in enumerate(hits[offset : offset + k]):
                doc_id = str(hit.get("document_id", ""))
                docs.append(
                    Document(
                        page_content=hit.get("text", ""),
                        metadata={
                            "document_id": doc_id,
                            "retrieval_rank": offset + local_rank,
                            "retrieval_offset": offset,
                            "retrieval_score": hit.get("_score", hit.get("_distance")),
                            "retrieval_source": retrieval_source,
                        },
                    )
                )
            out[offset] = docs[:k]
        return out

    def random_sample(self, k: int) -> list[Document]:
        if self.table is None:
            return []
        rows = self._random_sample_rows()
        if len(rows) <= k:
            sample = rows
        else:
            sample = random.sample(rows, k)

        return [
            Document(
                page_content=row.get("text", ""),
                metadata={
                    "document_id": str(row.get("document_id", "")),
                    "retrieval_rank": rank,
                    "retrieval_offset": None,
                    "retrieval_score": None,
                    "retrieval_source": "random",
                },
            )
            for rank, row in enumerate(sample)
        ]

    def _embed_query_cached(self, query_text: str):
        if self.query_vector_cache_size <= 0:
            return self.dense_embeddings.embed_query(query_text)

        with self._query_vector_cache_lock:
            vector = self._query_vector_cache.get(query_text)
            if vector is not None:
                self._query_vector_cache.move_to_end(query_text)
                return vector

        vector = self.dense_embeddings.embed_query(query_text)

        with self._query_vector_cache_lock:
            self._query_vector_cache[query_text] = vector
            self._query_vector_cache.move_to_end(query_text)
            while len(self._query_vector_cache) > self.query_vector_cache_size:
                self._query_vector_cache.popitem(last=False)

        return vector

    def _random_sample_rows(self):
        with self._random_sample_rows_lock:
            if self._random_sample_rows_cache is None:
                self._random_sample_rows_cache = self.table.to_arrow().select(["document_id", "text"]).to_pylist()
            return self._random_sample_rows_cache

    def _clear_query_vector_cache(self) -> None:
        with self._query_vector_cache_lock:
            self._query_vector_cache.clear()

    def _clear_random_sample_cache(self) -> None:
        with self._random_sample_rows_lock:
            self._random_sample_rows_cache = None

    def existing_document_ids(self) -> set[str]:
        if self.table is None:
            return set()

        existing: set[str] = set()
        for batch in self._document_id_batches():
            existing.update(str(doc_id) for doc_id in self._batch_document_ids(batch) if doc_id is not None)
        return existing

    @staticmethod
    def _batch_document_ids(batch) -> list:
        if hasattr(batch, "column_names") and "document_id" in batch.column_names:
            return batch.column(batch.column_names.index("document_id")).to_pylist()
        if hasattr(batch, "schema"):
            index = batch.schema.get_field_index("document_id")
            if index >= 0:
                return batch.column(index).to_pylist()
        return batch.column("document_id").to_pylist()

    def _document_id_batches(self):
        try:
            dataset = self.table.to_lance()
            if hasattr(dataset, "to_batches"):
                try:
                    yield from dataset.to_batches(columns=["document_id"], batch_size=100_000)
                    return
                except TypeError:
                    yield from dataset.to_batches(columns=["document_id"])
                    return
        except Exception:
            pass

        yield from self.table.to_arrow().select(["document_id"]).to_batches(max_chunksize=100_000)

    def _tune_query(self, query):
        if self.nprobes and hasattr(query, "nprobes"):
            query = query.nprobes(self.nprobes)
        if self.refine_factor and hasattr(query, "refine_factor"):
            query = query.refine_factor(self.refine_factor)
        return query

    def ensure_indexes(self) -> None:
        if self.table is None or self.count() == 0:
            return

        existing = []
        try:
            existing = list(self.table.list_indices())
        except Exception:
            existing = []

        index_text = "\n".join(repr(index).lower() for index in existing)
        has_vector_index = "vector" in index_text or "ivf" in index_text or "hnsw" in index_text
        has_fts_index = "fts" in index_text or "text" in index_text

        if not has_vector_index:
            try:
                print(
                    "Creating LanceDB vector index "
                    f"(IVF_FLAT, metric=cosine, target_partition_size={self.target_partition_size})..."
                )
                self.table.create_index(
                    metric="cosine",
                    vector_column_name="vector",
                    index_type="IVF_FLAT",
                    replace=False,
                    target_partition_size=self.target_partition_size,
                )
                print("LanceDB vector index created.")
            except Exception as exc:
                print(f"WARNING: LanceDB vector index was not created ({type(exc).__name__}: {exc})")

        if not has_fts_index:
            try:
                print("Creating LanceDB FTS index on 'text'...")
                self.table.create_fts_index("text", replace=False)
                print("LanceDB FTS index created.")
            except Exception as exc:
                print(f"WARNING: LanceDB FTS index was not created ({type(exc).__name__}: {exc})")

        try:
            print("LanceDB indices:")
            for index in self.table.list_indices():
                print(f"  - {index}")
        except Exception as exc:
            print(f"WARNING: LanceDB indices could not be listed ({type(exc).__name__}: {exc})")

    def optimize(self) -> None:
        if self.table is None or self.count() == 0:
            return

        print("Optimizing LanceDB table (compaction + pruning old versions)...")
        self.table.optimize(cleanup_older_than=timedelta(days=0), delete_unverified=False)
        try:
            self.table.checkout_latest()
        except Exception:
            self.table = self.db.open_table(self.collection_name)
        print("LanceDB optimize completed.")

    def compact_existing(self, mode: str = "none") -> None:
        if mode == "none" or self.table is None:
            return
        if mode == "optimize":
            self.optimize()
            return
        if mode != "rebuild":
            raise ValueError("--compact_existing must be one of: none, optimize, rebuild")

        self._rebuild_table_safely()

    def _rebuild_table_safely(self) -> None:
        table_dir = self._table_dir(self.collection_name)
        if not table_dir.exists():
            print(f"LanceDB table directory not found for rebuild: {table_dir}")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tmp_name = f"{self.collection_name}__rebuild_{timestamp}"
        tmp_dir = self._table_dir(tmp_name)
        backup_dir = table_dir.with_name(f"{table_dir.name}.backup-{timestamp}")

        if tmp_dir.exists() or backup_dir.exists():
            raise FileExistsError(f"Rebuild target already exists: {tmp_dir} or {backup_dir}")

        print(f"Rebuilding LanceDB table {self.collection_name!r} into compact table {tmp_name!r}...")
        data = self.table.to_arrow()
        expected_count = data.num_rows
        expected_ids = set(str(doc_id) for doc_id in data.column("document_id").to_pylist() if doc_id is not None)

        rebuilt = self.db.create_table(tmp_name, data=data, mode="overwrite")
        rebuilt_count = rebuilt.count_rows()
        if rebuilt_count != expected_count:
            raise RuntimeError(f"Rebuilt table row count mismatch: {rebuilt_count} != {expected_count}")

        try:
            self.table = None
            shutil.move(str(table_dir), str(backup_dir))
            shutil.move(str(tmp_dir), str(table_dir))
            self._reconnect()
            actual_count = self.count()
            actual_ids = self.existing_document_ids()
            if actual_count != expected_count or actual_ids != expected_ids:
                raise RuntimeError(
                    "Rebuilt table validation failed: "
                    f"count {actual_count}/{expected_count}, ids {len(actual_ids)}/{len(expected_ids)}"
                )
        except Exception:
            if table_dir.exists() and backup_dir.exists():
                failed_dir = table_dir.with_name(f"{table_dir.name}.failed-rebuild-{timestamp}")
                shutil.move(str(table_dir), str(failed_dir))
            if backup_dir.exists() and not table_dir.exists():
                shutil.move(str(backup_dir), str(table_dir))
            self._reconnect()
            raise

        print(f"LanceDB rebuild completed. Old table backup: {backup_dir}")

    def _reconnect(self) -> None:
        import lancedb

        self.db = lancedb.connect(self.lancedb_path)
        self.table = self._load_table_if_exists(self.collection_name)
        self._clear_query_vector_cache()
        self._clear_random_sample_cache()

    def _table_dir(self, collection_name: str) -> Path:
        return Path(self.lancedb_path) / f"{collection_name}.lance"

    def print_storage_stats(self) -> None:
        root = self._table_dir(self.collection_name)
        if not root.exists():
            return

        print("LanceDB storage stats:")
        for rel in ("data", "_versions", "_transactions"):
            path = root / rel
            if path.exists():
                size = self._directory_size(path)
                files = sum(1 for item in path.rglob("*") if item.is_file())
                print(f"  {rel}: {files} files, {self._format_bytes(size)}")
        print(f"  total: {self._format_bytes(self._directory_size(root))}")

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += (Path(dirpath) / filename).stat().st_size
                except OSError:
                    pass
        return total

    @staticmethod
    def _format_bytes(size: int) -> str:
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TiB"


def create_vector_backend(
    collection_name: str,
    dense_embeddings,
    sparse_embeddings=None,
    dense_dim_size: int | None = None,
) -> VectorBackend:
    backend = get_vector_db_backend()

    if backend == "qdrant":
        if sparse_embeddings is None or dense_dim_size is None:
            raise ValueError("Qdrant backend requires sparse embeddings and dense_dim_size")
        return QdrantBackend(
            collection_name=collection_name,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            dense_dim_size=dense_dim_size,
        )

    return LanceDBBackend(collection_name=collection_name, dense_embeddings=dense_embeddings)
