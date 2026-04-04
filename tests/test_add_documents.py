import sys
from pathlib import Path

from datasets import Dataset
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from add_documents_to_db import add_documents_from_dataset, filter_dataset_by_missing_ids_ds


class FakeBackend:
    def __init__(self):
        self.added = []

    def upsert_documents(self, documents):
        self.added.extend(documents)


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
