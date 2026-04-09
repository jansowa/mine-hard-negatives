from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Iterable

from langchain_core.documents import Document

from config import get_lancedb_path, get_qdrant_url, get_vector_db_backend


class VectorBackend(ABC):
    @abstractmethod
    def upsert_documents(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def search(self, query_text: str, k: int, offset: int = 0) -> list[Document]:
        ...

    @abstractmethod
    def random_sample(self, k: int) -> list[Document]:
        ...

    @abstractmethod
    def existing_document_ids(self) -> set[str]:
        ...


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
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
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
        return self.vector_store.similarity_search(query=query_text, k=k, offset=offset)

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
    def __init__(self, collection_name: str, dense_embeddings, lancedb_path: str | None = None) -> None:
        import lancedb

        self.collection_name = collection_name
        self.dense_embeddings = dense_embeddings
        self.db = lancedb.connect(lancedb_path or get_lancedb_path())
        self.table = self._load_table_if_exists(collection_name)

    def _load_table_if_exists(self, collection_name: str):
        table_names = set(self.db.table_names())
        if collection_name in table_names:
            return self.db.open_table(collection_name)
        return None

    def _ensure_dimensionality(self, vector: Iterable[float]) -> list[float]:
        values = list(vector)
        if self.count() == 0:
            return values
        if len(values) == 0:
            raise ValueError("Dense embedding returned empty vector")
        return values

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [d.page_content for d in documents]
        vectors = self.dense_embeddings.embed_documents(texts)

        rows = []
        for doc, vector in zip(documents, vectors):
            rows.append(
                {
                    "document_id": str(doc.metadata.get("document_id")),
                    "text": doc.page_content,
                    "vector": self._ensure_dimensionality(vector),
                }
            )

        if self.table is None:
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

        vector = self.dense_embeddings.embed_query(query_text)
        limit = max(k + offset, k)

        # Prefer hybrid query: dense + BM25 FTS.
        try:
            hits = self.table.search(query_type="hybrid", query=query_text, vector=vector).limit(limit).to_list()
        except Exception:
            hits = self.table.search(vector).limit(limit).to_list()

        docs = []
        for hit in hits[offset:offset + k]:
            doc_id = str(hit.get("document_id", ""))
            docs.append(Document(page_content=hit.get("text", ""), metadata={"document_id": doc_id}))

        return docs[:k]

    def random_sample(self, k: int) -> list[Document]:
        if self.table is None:
            return []
        rows = self.table.to_list()
        if len(rows) <= k:
            sample = rows
        else:
            sample = random.sample(rows, k)

        return [Document(page_content=row.get("text", ""), metadata={"document_id": str(row.get("document_id", ""))}) for row in sample]

    def existing_document_ids(self) -> set[str]:
        if self.table is None:
            return set()
        return {
            str(row.get("document_id"))
            for row in self.table.to_list()
        }


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
