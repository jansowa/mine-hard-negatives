import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from multi_gpu_processor import GPUModelSet


class FakeBackend:
    def __init__(self):
        self.search_calls = 0

    def search(self, query_text, k, offset=0):
        self.search_calls += 1
        return [
            Document(page_content="doc1", metadata={"document_id": "d1"}),
            Document(page_content="doc2", metadata={"document_id": "d2"}),
        ]

    def random_sample(self, k):
        return [Document(page_content="doc3", metadata={"document_id": "d3"})]


def _build_model_set_for_test():
    model_set = GPUModelSet.__new__(GPUModelSet)
    model_set.reranker_tokenizer = object()
    model_set.reranker_model = object()
    model_set.total_queries = 0
    model_set.beta = 0.4
    model_set.u_floor = 0.1
    model_set._ecdf_x = [0.0, 0.5, 1.0]
    model_set._ecdf_y = [0.0, 0.5, 1.0]
    return model_set


def test_process_query_batch_uses_backend_and_collects_docs():
    model_set = _build_model_set_for_test()
    backend = FakeBackend()

    def fake_rerank(*args, **kwargs):
        return [0.2, 0.3]

    results = model_set.process_query_batch(
        [
            {
                "query_id": "q1",
                "document_id": "positive",
                "positive_ranking": 0.9,
                "text": "query",
            }
        ],
        backend,
        fake_rerank,
        reranker_batch_size=8,
    )

    assert backend.search_calls >= 1
    assert len(results) >= 2
    assert all(row[0] == "q1" for row in results)
