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
            Document(
                page_content="doc1",
                metadata={
                    "document_id": "d1",
                    "retrieval_rank": offset,
                    "retrieval_offset": offset,
                    "retrieval_score": 0.9,
                    "retrieval_source": "test",
                },
            ),
            Document(
                page_content="doc2",
                metadata={
                    "document_id": "d2",
                    "retrieval_rank": offset + 1,
                    "retrieval_offset": offset,
                    "retrieval_score": 0.8,
                    "retrieval_source": "test",
                },
            ),
        ]

    def random_sample(self, k):
        return []


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
    assert all(row["query_id"] == "q1" for row in results)


def test_process_query_batch_marks_candidates_without_dropping_unselected():
    model_set = _build_model_set_for_test()
    backend = FakeBackend()

    def fake_rerank(*args, **kwargs):
        return [0.2, 0.8]

    results = model_set.process_query_batch(
        [
            {
                "query_id": "q1",
                "document_id": "positive",
                "positive_ranking": 1.0,
                "text": "query",
            }
        ],
        backend,
        fake_rerank,
        reranker_batch_size=8,
    )

    selected_by_doc = {row["document_id"]: row["candidate_selected"] for row in results}
    assert selected_by_doc == {"d1": True, "d2": False}
    assert all("candidate_percentile" in row for row in results)
    assert all(row["retrieval_source"] == "test" for row in results)


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


def test_consolidate_worker_files_supports_candidate_schema(tmp_path):
    output_path = tmp_path / "negative_candidates.parquet"
    processor = MultiGPUNegativeFinder(
        [_build_processor_model_set_for_test()],
        output_path=str(output_path),
        ranking_column="candidate_ranking",
        profile_timing=True,
    )
    processor.parquet_row_group_size = 2

    rows = [
        {
            "query_id": "q1",
            "document_id": "d1",
            "candidate_ranking": 0.1,
            "candidate_percentile": 0.2,
            "candidate_selected": True,
            "retrieval_rank": 0,
            "retrieval_offset": 0,
            "retrieval_score": 0.9,
            "retrieval_source": "hybrid",
        },
        {
            "query_id": "q1",
            "document_id": "d2",
            "candidate_ranking": 0.8,
            "candidate_percentile": 0.9,
            "candidate_selected": False,
            "retrieval_rank": 1,
            "retrieval_offset": 0,
            "retrieval_score": 0.7,
            "retrieval_source": "hybrid",
        },
    ]
    with open(processor.worker_files[0], "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    processor.consolidate_worker_files()

    df = pd.read_parquet(output_path)
    assert set(df.columns) == {
        "query_id",
        "document_id",
        "candidate_ranking",
        "candidate_percentile",
        "candidate_selected",
        "retrieval_rank",
        "retrieval_offset",
        "retrieval_score",
        "retrieval_source",
    }
    assert df["candidate_selected"].tolist() == [True, False]
