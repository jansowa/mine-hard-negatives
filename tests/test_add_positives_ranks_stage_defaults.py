import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

import add_positives_ranks as apr
from add_positives_ranks import get_positive_ranks_stage_defaults


def test_positive_ranks_candidate_stage_uses_candidate_env(monkeypatch):
    monkeypatch.setenv("CANDIDATE_POSITIVE_RANKS_OUTPUT_PATH", "data/relevant_candidate.parquet")
    monkeypatch.setenv("CANDIDATE_RERANKER_NAME", "small-reranker")
    monkeypatch.setenv("CANDIDATE_RERANKER_BATCH_SIZE", "17")
    monkeypatch.setenv("CANDIDATE_POSITIVE_RANKS_SCORE_COLUMN", "positive_candidate_ranking")

    defaults = get_positive_ranks_stage_defaults(final_step=False)

    assert defaults == {
        "output_path": "data/relevant_candidate.parquet",
        "reranker_model_name": "small-reranker",
        "reranker_batch_size": 17,
        "score_column": "positive_candidate_ranking",
    }


def test_positive_ranks_final_stage_uses_final_env(monkeypatch):
    monkeypatch.setenv("FINAL_POSITIVE_RANKS_OUTPUT_PATH", "data/relevant_final.parquet")
    monkeypatch.setenv("FINAL_RERANKER_NAME", "large-reranker")
    monkeypatch.setenv("FINAL_RERANKER_BATCH_SIZE", "5")
    monkeypatch.setenv("FINAL_POSITIVE_RANKS_SCORE_COLUMN", "positive_ranking")

    defaults = get_positive_ranks_stage_defaults(final_step=True)

    assert defaults == {
        "output_path": "data/relevant_final.parquet",
        "reranker_model_name": "large-reranker",
        "reranker_batch_size": 5,
        "score_column": "positive_ranking",
    }


def test_process_relevant_auto_tunes_reranker_batch_size(monkeypatch, tmp_path):
    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    output_path = tmp_path / "relevant_with_score.parquet"

    pd.DataFrame({"id": ["q1", "q2"], "text": ["query one", "query two"]}).to_parquet(queries_path, index=False)
    pd.DataFrame({"id": ["d1", "d2"], "text": ["doc one", "doc two"]}).to_parquet(corpus_path, index=False)
    pd.DataFrame({"query_id": ["q1", "q2"], "document_id": ["d1", "d2"]}).to_parquet(relevant_path, index=False)

    monkeypatch.setattr(apr, "get_reranker_model", lambda _model_name: (object(), object()))
    monkeypatch.setattr(
        apr,
        "build_or_load_corpus_sqlite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SQLite must require explicit low-memory opt-in")),
    )
    benchmark_calls = []

    def fake_benchmark(_tokenizer, _model, sample_queries, sample_docs, candidates, _rerank_function, **_kwargs):
        benchmark_calls.append((sample_queries, sample_docs, candidates))
        return 3

    monkeypatch.setattr(apr, "benchmark_reranker_batch_size", fake_benchmark)
    rerank_batch_sizes = []

    def fake_rerank(_tokenizer, _model, _queries, docs, batch_size, model_name):
        rerank_batch_sizes.append((batch_size, model_name))
        return [0.5] * len(docs)

    monkeypatch.setattr(apr, "rerank", fake_rerank)

    apr.process_relevant(
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        relevant_path=str(relevant_path),
        output_path=str(output_path),
        chunk_size=2,
        reranker_batch_size=None,
        reranker_model_name="positive-model",
        auto_reranker_batch_size_candidates="1,3",
        auto_reranker_batch_size_sample_size=2,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
    )

    df = pd.read_parquet(output_path)
    assert df["query_id"].tolist() == ["q1", "q2"]
    assert benchmark_calls == [(["query one", "query two"], ["doc one", "doc two"], [1, 3])]
    assert rerank_batch_sizes == [(3, "positive-model")]


def test_process_relevant_preserves_extra_relevant_columns(monkeypatch, tmp_path):
    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    output_path = tmp_path / "relevant_with_score.parquet"

    pd.DataFrame({"id": ["q1"], "text": ["query one"]}).to_parquet(queries_path, index=False)
    pd.DataFrame({"id": ["d1"], "text": ["doc one"]}).to_parquet(corpus_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["d1"],
            "lightonai_positive_score": [0.75],
        }
    ).to_parquet(relevant_path, index=False)

    monkeypatch.setattr(apr, "get_reranker_model", lambda _model_name: (object(), object()))
    monkeypatch.setattr(apr, "rerank", lambda *_args, **_kwargs: [1.25])

    apr.process_relevant(
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        relevant_path=str(relevant_path),
        output_path=str(output_path),
        chunk_size=2,
        reranker_batch_size=1,
        reranker_model_name="positive-model",
        score_column="positive_ranking",
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        low_memory_optimizations=True,
    )

    df = pd.read_parquet(output_path)
    assert df.to_dict("list") == {
        "query_id": ["q1"],
        "document_id": ["d1"],
        "lightonai_positive_score": [0.75],
        "positive_ranking": [1.25],
    }


def test_process_relevant_expanding_limit_scores_only_missing_pairs(monkeypatch, tmp_path):
    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    output_path = tmp_path / "relevant_with_score.parquet"

    pd.DataFrame({"id": ["q1", "q2"], "text": ["query one", "query two"]}).to_parquet(queries_path, index=False)
    pd.DataFrame({"id": ["d1", "d2"], "text": ["doc one", "doc two"]}).to_parquet(corpus_path, index=False)
    pd.DataFrame({"query_id": ["q1", "q2"], "document_id": ["d1", "d2"]}).to_parquet(relevant_path, index=False)

    monkeypatch.setattr(apr, "get_reranker_model", lambda _model_name: (object(), object()))
    calls = []

    def fake_rerank(_tokenizer, _model, _queries, docs, batch_size, model_name):
        calls.append(list(docs))
        return [1.0] * len(docs)

    monkeypatch.setattr(apr, "rerank", fake_rerank)
    kwargs = {
        "queries_path": str(queries_path),
        "corpus_path": str(corpus_path),
        "relevant_path": str(relevant_path),
        "output_path": str(output_path),
        "chunk_size": 2,
        "reranker_batch_size": 2,
        "reranker_model_name": "positive-model",
        "score_column": "positive_ranking",
    }

    apr.process_relevant(**kwargs, offset=1)
    apr.process_relevant(**kwargs, offset=None)

    assert calls == [["doc one"], ["doc two"]]
    assert set(pd.read_parquet(output_path)["document_id"]) == {"d1", "d2"}
