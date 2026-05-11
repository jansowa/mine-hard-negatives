import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

import rerank_negative_candidates as rnc


def _write_inputs(tmp_path):
    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    candidates_path = tmp_path / "negative_candidates.parquet"

    pd.DataFrame(
        {
            "id": ["q1", "q2"],
            "text": ["query one", "query two"],
        }
    ).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "id": ["d1", "d2", "d3"],
            "text": ["doc one", "doc two", "doc three"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "document_id": ["d1", "d2", "d3"],
            "candidate_ranking": [0.1, 0.9, 0.2],
            "candidate_percentile": [0.1, 0.9, 0.2],
            "candidate_selected": [True, False, True],
            "retrieval_rank": [0, 1, 0],
            "retrieval_offset": [0, 0, 128],
            "retrieval_score": [0.9, 0.8, 0.7],
            "retrieval_source": ["hybrid", "hybrid", "hybrid"],
        }
    ).to_parquet(candidates_path, index=False)

    return queries_path, corpus_path, candidates_path


def test_rerank_candidates_scores_selected_and_writes_ranking_alias(monkeypatch, tmp_path):
    queries_path, corpus_path, candidates_path = _write_inputs(tmp_path)
    output_path = tmp_path / "negatives.parquet"
    calls = []

    monkeypatch.setattr(rnc, "get_reranker_model", lambda _model_name: (object(), object()))

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size, model_name):
        calls.append((queries, docs, batch_size, model_name))
        return [10.0 + idx for idx, _ in enumerate(docs)]

    monkeypatch.setattr(rnc, "rerank", fake_rerank)

    rnc.rerank_candidates(
        candidates_path=str(candidates_path),
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        output_path=str(output_path),
        reranker_model_name="final-model",
        reranker_batch_size=2,
        chunk_size=2,
        row_group_size=1,
        resume=True,
    )

    df = pd.read_parquet(output_path)
    assert df["document_id"].tolist() == ["d1", "d3"]
    assert "candidate_ranking" in df.columns
    assert "final_ranking" in df.columns
    assert "ranking" in df.columns
    assert df["final_ranking"].tolist() == df["ranking"].tolist()
    assert len(calls) == 2
    assert calls[0][2] == 2
    assert calls[0][3] == "final-model"


def test_rerank_candidates_auto_tunes_reranker_batch_size(monkeypatch, tmp_path):
    queries_path, corpus_path, candidates_path = _write_inputs(tmp_path)
    output_path = tmp_path / "negatives.parquet"
    benchmark_calls = []
    rerank_calls = []

    monkeypatch.setattr(rnc, "get_reranker_model", lambda _model_name: (object(), object()))

    def fake_benchmark(_tokenizer, _model, sample_queries, sample_docs, candidates, _rerank_function, **_kwargs):
        benchmark_calls.append((sample_queries, sample_docs, candidates))
        return 4

    monkeypatch.setattr(rnc, "benchmark_reranker_batch_size", fake_benchmark)

    def fake_rerank(_tokenizer, _model, _queries, docs, batch_size, model_name):
        rerank_calls.append((docs, batch_size, model_name))
        return [1.0] * len(docs)

    monkeypatch.setattr(rnc, "rerank", fake_rerank)

    rnc.rerank_candidates(
        candidates_path=str(candidates_path),
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        output_path=str(output_path),
        reranker_model_name="final-model",
        reranker_batch_size=None,
        chunk_size=3,
        auto_reranker_batch_size_candidates="2,4",
        auto_reranker_batch_size_sample_size=2,
    )

    assert benchmark_calls == [(["query one", "query two"], ["doc one", "doc three"], [2, 4])]
    assert rerank_calls == [(["doc one", "doc three"], 4, "final-model")]


def test_rerank_candidates_resume_skips_already_scored_pairs(monkeypatch, tmp_path):
    queries_path, corpus_path, candidates_path = _write_inputs(tmp_path)
    output_path = tmp_path / "negatives.parquet"
    calls = []

    monkeypatch.setattr(rnc, "get_reranker_model", lambda _model_name: (object(), object()))

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size, model_name):
        calls.append(docs)
        return [1.0] * len(docs)

    monkeypatch.setattr(rnc, "rerank", fake_rerank)

    kwargs = {
        "candidates_path": str(candidates_path),
        "queries_path": str(queries_path),
        "corpus_path": str(corpus_path),
        "output_path": str(output_path),
        "reranker_model_name": "final-model",
        "reranker_batch_size": 2,
        "chunk_size": 3,
        "resume": True,
    }
    rnc.rerank_candidates(**kwargs)
    assert len(calls) == 1

    def fail_rerank(*_args, **_kwargs):
        raise AssertionError("resume should skip all existing pairs")

    monkeypatch.setattr(rnc, "rerank", fail_rerank)
    rnc.rerank_candidates(**kwargs)

    df = pd.read_parquet(output_path)
    assert df["document_id"].tolist() == ["d1", "d3"]


def test_rerank_candidates_adaptive_expands_only_until_target(monkeypatch, tmp_path):
    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    candidates_path = tmp_path / "negative_candidates.parquet"
    output_path = tmp_path / "negatives.parquet"
    report_path = tmp_path / "report.json"

    pd.DataFrame(
        {
            "id": ["q1", "q2"],
            "text": ["query one", "query two"],
        }
    ).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "id": ["d1", "d2", "d3", "d4", "d5", "d6", "p1", "p2"],
            "text": ["d1", "d2", "d3", "d4", "d5", "d6", "p1", "p2"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "document_id": ["p1", "p2"],
            "positive_ranking": [1.0, 1.0],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "document_id": ["d1", "d2", "d3", "d4", "d5", "d6"],
            "candidate_ranking": [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
            "candidate_percentile": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "candidate_selected": [True, True, False, True, False, False],
            "retrieval_rank": [0, 1, 2, 0, 1, 2],
            "retrieval_offset": [0, 0, 0, 0, 0, 0],
            "retrieval_score": [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
            "retrieval_source": ["hybrid"] * 6,
        }
    ).to_parquet(candidates_path, index=False)

    monkeypatch.setattr(rnc, "get_reranker_model", lambda _model_name: (object(), object()))

    final_scores = {"d1": 0.5, "d2": 0.5, "d3": 0.5, "d4": 2.0, "d5": 0.5, "d6": 0.5}
    calls = []

    def fake_rerank(_tokenizer, _model, _queries, docs, batch_size, model_name):
        calls.append(list(docs))
        return [final_scores[doc] for doc in docs]

    monkeypatch.setattr(rnc, "rerank", fake_rerank)

    rnc.rerank_candidates(
        candidates_path=str(candidates_path),
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        relevant_path=str(relevant_path),
        output_path=str(output_path),
        reranker_model_name="final-model",
        reranker_batch_size=2,
        ranking_column="final_ranking",
        rerank_mode="adaptive",
        positive_score_column="positive_ranking",
        num_negatives=2,
        beta=1.0,
        u_floor=0.0,
        initial_budget=1,
        budget_step=1,
        max_budget=3,
        resume=False,
        report_path=str(report_path),
    )

    df = pd.read_parquet(output_path)
    by_query = {query_id: set(group["document_id"]) for query_id, group in df.groupby("query_id")}
    assert by_query == {"q1": {"d1", "d2"}, "q2": {"d4", "d5", "d6"}}
    scored_docs = [doc for call in calls for doc in call]
    assert all("d3" not in call for call in calls)
    assert scored_docs.index("d6") < scored_docs.index("d5")
    assert set(df.columns) >= {"final_selected", "final_threshold_rank", "final_rerank_budget"}
