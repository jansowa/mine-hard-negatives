import sys
import threading
from collections import OrderedDict
from pathlib import Path

from datasets import Dataset
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from add_documents_to_db import (
    add_documents_from_dataset,
    add_documents_to_lancedb_from_dataset,
    filter_dataset_by_missing_ids_ds,
)
from utils.vector_db import LanceDBBackend


class FakeBackend:
    def __init__(self):
        self.added = []
        self.calls = []

    def upsert_documents(self, documents):
        self.calls.append(list(documents))
        self.added.extend(documents)


class FakeLanceBackend:
    def __init__(self):
        self.calls = []

    def upsert_embeddings(self, document_ids, texts, vectors):
        self.calls.append((list(document_ids), list(texts), vectors))


class FakeSentenceTransformerClient:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, texts, show_progress_bar=False, **kwargs):
        import numpy as np

        self.batch_sizes.append(kwargs["batch_size"])
        return np.zeros((len(texts), 2), dtype=np.float32)


class FakeDenseEmbeddings:
    def __init__(self, batch_size):
        self.encode_kwargs = {"batch_size": batch_size}
        self.show_progress = False
        self._client = FakeSentenceTransformerClient()


class FakeQueryEmbeddings:
    def __init__(self):
        self.encode_kwargs = {"prompt": "doc: "}
        self.query_encode_kwargs = {"prompt": "query: "}
        self.calls = []

    def _embed(self, texts, encode_kwargs):
        self.calls.append((list(texts), dict(encode_kwargs)))
        return [[float(index), float(index + 1)] for index, _ in enumerate(texts)]

    def embed_query(self, _text):
        raise AssertionError("cached query vector should be reused")


def _build_lancedb_backend_for_query_cache(dense_embeddings):
    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.dense_embeddings = dense_embeddings
    backend.query_vector_cache_size = 10
    backend._query_vector_cache = OrderedDict()
    backend._query_vector_cache_lock = threading.Lock()
    return backend


def test_filter_dataset_by_missing_ids_ds_removes_existing_ids():
    ds = Dataset.from_dict({"id": ["1", "2", "3"], "text": ["a", "b", "c"]})
    filtered = filter_dataset_by_missing_ids_ds(ds, {"2"}, num_proc=1, batch_size=2)

    assert filtered["id"] == ["1", "3"]


def test_add_documents_from_dataset_skips_empty_content():
    ds = Dataset.from_dict({"id": ["1", "2"], "text": ["valid", ""]})
    backend = FakeBackend()

    add_documents_from_dataset(ds, batch_size=10, backend=backend)

    assert len(backend.added) == 1
    assert isinstance(backend.added[0], Document)
    assert backend.added[0].metadata["document_id"] == "1"


def test_add_documents_from_dataset_uses_db_write_batch_size():
    ds = Dataset.from_dict({"id": [str(i) for i in range(5)], "text": [f"text {i}" for i in range(5)]})
    backend = FakeBackend()

    add_documents_from_dataset(ds, batch_size=1, db_write_batch_size=2, backend=backend)

    assert [len(call) for call in backend.calls] == [2, 2, 1]


def test_lancedb_fast_path_keeps_write_batches_larger_than_gpu_microbatches():
    ds = Dataset.from_dict({"id": [str(i) for i in range(5)], "text": [f"text {i}" for i in range(5)]})
    embeddings = FakeDenseEmbeddings(batch_size=2)
    backend = FakeLanceBackend()

    add_documents_to_lancedb_from_dataset(
        ds,
        embeddings,
        backend,
        db_write_batch_size=3,
        async_write=False,
    )

    assert [len(call[0]) for call in backend.calls] == [3, 2]
    assert embeddings._client.batch_sizes == [2, 2]


def test_lancedb_prepare_query_vectors_batches_missing_queries_into_cache():
    embeddings = FakeQueryEmbeddings()
    backend = _build_lancedb_backend_for_query_cache(embeddings)

    backend.prepare_query_vectors(["q1", "q2", "q1"])

    assert embeddings.calls == [(["q1", "q2"], {"prompt": "query: "})]
    assert list(backend._query_vector_cache) == ["q1", "q2"]
    assert backend._embed_query_cached("q1") == [0.0, 1.0]
    assert backend._embed_query_cached("q2") == [1.0, 2.0]


def test_normalise_vectors_does_not_count_rows_per_document():
    backend = LanceDBBackend.__new__(LanceDBBackend)
    backend.count = lambda: (_ for _ in ()).throw(AssertionError("count_rows should not be called"))

    assert backend._normalise_vectors([[1.0, 2.0], [3.0, 4.0]]) == [[1.0, 2.0], [3.0, 4.0]]


def test_normalise_vectors_rejects_inconsistent_dimensions():
    backend = LanceDBBackend.__new__(LanceDBBackend)

    try:
        backend._normalise_vectors([[1.0, 2.0], [3.0]])
    except ValueError as exc:
        assert "Inconsistent dense embedding dimensions" in str(exc)
    else:
        raise AssertionError("Expected inconsistent dimensions to be rejected")
