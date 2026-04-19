import sys
from pathlib import Path
import json

import pandas as pd

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from multi_gpu_processor import GPUModelSet, MultiGPUNegativeFinder


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


def _build_processor_model_set_for_test():
    model_set = GPUModelSet.__new__(GPUModelSet)
    model_set.gpu_id = 0
    model_set.total_queries = 0
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


def test_consolidate_worker_files_streams_to_parquet(tmp_path):
    output_path = tmp_path / "negatives.parquet"
    processor = MultiGPUNegativeFinder(
        [_build_processor_model_set_for_test()],
        output_path=str(output_path),
        profile_timing=True,
    )
    processor.parquet_row_group_size = 2

    rows = [
        {"query_id": "q1", "document_id": "d1", "ranking": 0.1},
        {"query_id": "q1", "document_id": "d2", "ranking": 0.2},
        {"query_id": "q2", "document_id": "d3", "ranking": 0.3},
    ]
    with open(processor.worker_files[0], "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    processor.consolidate_worker_files()

    df = pd.read_parquet(output_path)
    got = {
        (row.query_id, row.document_id, round(float(row.ranking), 6))
        for row in df.itertuples(index=False)
    }
    expected = {
        (row["query_id"], row["document_id"], round(float(row["ranking"]), 6))
        for row in rows
    }
    assert got == expected
