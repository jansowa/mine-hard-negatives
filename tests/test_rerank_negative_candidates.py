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
