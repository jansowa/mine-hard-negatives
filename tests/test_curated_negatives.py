import json
import sys
from pathlib import Path

import datasets
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from curated_negatives.lightonai_to_flag_embedding import LightOnKDToFlagEmbedding, _load_lightonai_component
from curated_negatives.lightonai_to_pipeline_artifacts import write_lightonai_pipeline_artifacts
from curated_negatives.run_lightonai_adaptive_pipeline import (
    _parse_stages,
    run_lightonai_pipeline,
    split_paths,
)
from curated_negatives.score_flag_embedding_jsonl import incomplete_output_path, score_flag_embedding_jsonl


def _jsonl_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_lightonai_processor_maps_to_flag_embedding_row():
    queries = datasets.Dataset.from_list([{"query_id": 10, "query": "what is retrieval?"}])
    documents = datasets.Dataset.from_list(
        [
            {"document_id": 1, "document": "positive passage"},
            {"document_id": 2, "document": "first negative"},
            {"document_id": 3, "document": "second negative"},
            {"document_id": 4, "document": "too close to positive"},
        ]
    )
    processor = LightOnKDToFlagEmbedding(
        queries,
        documents,
        num_negatives=2,
        nv_threshold=0.99,
        prompt="Represent this sentence for searching relevant passages:",
        dataset_type="retrieval",
    )
    example = {
        "query_id": 10,
        "document_ids": [1, 2, 3, 4],
        "scores": [1.0, 0.4, 0.8, 0.995],
    }

    assert processor.has_enough_negatives(example)
    row = processor.map_to_flag_embedding(example)

    assert row == {
        "query": "what is retrieval?",
        "pos": ["positive passage"],
        "neg": ["first negative", "second negative"],
        "pos_scores": [1.0],
        "neg_scores": [0.4, 0.8],
        "prompt": "Represent this sentence for searching relevant passages:",
        "type": "retrieval",
        "source_dataset": "lightonai/embeddings-fine-tuning",
        "query_id": 10,
        "pos_id": [1],
        "neg_id": [2, 3],
    }


def test_lightonai_pipeline_artifacts_are_compact_parquet_files(tmp_path):
    queries = datasets.Dataset.from_list(
        [
            {"query_id": 10, "query": "query ten"},
            {"query_id": 20, "query": "query twenty"},
        ]
    )
    documents = datasets.Dataset.from_list(
        [
            {"document_id": 1, "document": "positive one"},
            {"document_id": 2, "document": "negative two"},
            {"document_id": 3, "document": "negative three"},
        ]
    )
    scores = datasets.Dataset.from_list(
        [
            {"query_id": 10, "document_ids": [1, 2, 3], "scores": [0.9, 0.4, 0.6]},
            {"query_id": 20, "document_ids": [3, 2], "scores": [0.8, 0.1]},
        ]
    )

    write_lightonai_pipeline_artifacts(str(tmp_path), queries, documents, scores)

    queries_df = pd.read_parquet(tmp_path / "queries.parquet")
    corpus_df = pd.read_parquet(tmp_path / "corpus.parquet")
    relevant_df = pd.read_parquet(tmp_path / "relevant.parquet")
    candidates_df = pd.read_parquet(tmp_path / "negative_candidates.parquet")

    assert queries_df.to_dict("list") == {"id": ["10", "20"], "text": ["query ten", "query twenty"]}
    assert corpus_df.to_dict("list") == {
        "id": ["1", "2", "3"],
        "text": ["positive one", "negative two", "negative three"],
    }
    assert relevant_df.to_dict("list") == {
        "query_id": ["10", "20"],
        "document_id": ["1", "3"],
        "lightonai_positive_score": [0.9, 0.8],
    }
    assert candidates_df["query_id"].tolist() == ["10", "10", "20"]
    assert candidates_df["document_id"].tolist() == ["2", "3", "2"]
    assert candidates_df["candidate_ranking"].tolist() == [0.4, 0.6, 0.1]
    assert candidates_df["lightonai_score"].tolist() == [0.4, 0.6, 0.1]
    assert candidates_df["candidate_selected"].tolist() == [True, True, True]
    assert candidates_df["retrieval_rank"].tolist() == [0, 1, 0]
    assert candidates_df["retrieval_source"].tolist() == ["lightonai", "lightonai", "lightonai"]
    assert not {"query", "document", "text", "query_text", "document_text"} & set(candidates_df.columns)


def test_lightonai_loader_scopes_data_files_to_requested_split(monkeypatch):
    calls = []
    expected_dataset = datasets.Dataset.from_list([{"query_id": 1, "document_ids": [2], "scores": [0.5]}])

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return expected_dataset

    monkeypatch.setattr("curated_negatives.lightonai_to_flag_embedding.datasets.load_dataset", fake_load_dataset)

    loaded = _load_lightonai_component(
        dataset_name="lightonai/embeddings-fine-tuning",
        component_name="scores",
        split="fiqa",
        hf_cache_dir=None,
        load_num_proc=None,
    )

    assert loaded is expected_dataset
    assert calls == [
        {
            "path": "lightonai/embeddings-fine-tuning",
            "name": "scores",
            "split": "fiqa",
            "cache_dir": None,
            "data_files": {"fiqa": "scores/fiqa-*"},
            "verification_mode": "no_checks",
        }
    ]


def test_lightonai_split_paths_are_derived_from_split_name(tmp_path):
    paths = split_paths(str(tmp_path), "fiqa")

    assert paths.output_dir == str(tmp_path / "fiqa")
    assert paths.queries_path == str(tmp_path / "fiqa" / "queries.parquet")
    assert paths.corpus_path == str(tmp_path / "fiqa" / "corpus.parquet")
    assert paths.relevant_path == str(tmp_path / "fiqa" / "relevant.parquet")
    assert paths.relevant_with_score_path == str(tmp_path / "fiqa" / "relevant_with_score.parquet")
    assert paths.negative_candidates_path == str(tmp_path / "fiqa" / "negative_candidates.parquet")
    assert paths.negatives_path == str(tmp_path / "fiqa" / "negatives.parquet")
    assert paths.output_jsonl_path == str(tmp_path / "fiqa" / "train.jsonl")


def test_lightonai_stage_parser_keeps_pipeline_order():
    assert _parse_stages("jsonl,artifacts,negatives") == ["artifacts", "negatives", "jsonl"]


def test_lightonai_pipeline_runner_uses_env_defaults_and_split_dirs(monkeypatch, tmp_path):
    import curated_negatives.run_lightonai_adaptive_pipeline as runner

    calls = []

    monkeypatch.setenv("RERANKER_NAME", "mixedbread-ai/mxbai-rerank-base-v2")
    monkeypatch.setenv("FINAL_RERANKER_NAME", "mixedbread-ai/mxbai-rerank-base-v2")
    monkeypatch.setenv("NUM_NEGATIVES", "7")
    monkeypatch.setenv("FINAL_BETA", "0.01")
    monkeypatch.setenv("FINAL_U_FLOOR", "0.005")
    monkeypatch.setenv("FINAL_RERANK_INITIAL_BUDGET", "3")
    monkeypatch.setenv("FINAL_RERANK_BUDGET_STEP", "2")
    monkeypatch.setenv("FINAL_RERANK_MAX_BUDGET", "0")
    monkeypatch.setenv("PROCESSING_CHUNK_SIZE", "11")
    monkeypatch.setenv("QUERY_CHUNK_SIZE", "13")
    monkeypatch.setenv("OVERSAMPLE_FACTOR", "5")
    monkeypatch.setenv("MAX_NEG_REUSE", "17")
    monkeypatch.setenv("POSITIVE_ORIGINAL_SCORE_COLUMN", "lightonai_positive_score")
    monkeypatch.setenv("NEGATIVE_ORIGINAL_SCORE_COLUMN", "candidate_ranking")

    monkeypatch.setattr(runner, "export_lightonai_pipeline_artifacts", lambda **kwargs: calls.append(("artifacts", kwargs)))

    def fake_process_relevant(**kwargs):
        calls.append(("positives", kwargs))

    monkeypatch.setattr(runner, "process_relevant", fake_process_relevant)

    def fake_rerank_candidates(**kwargs):
        calls.append(("negatives", kwargs))

    monkeypatch.setattr(runner, "rerank_candidates", fake_rerank_candidates)

    def fake_process_negatives_streaming(**kwargs):
        calls.append(("jsonl", kwargs))

    monkeypatch.setattr(runner, "process_negatives_streaming", fake_process_negatives_streaming)

    run_lightonai_pipeline(
        splits=["fiqa", "nq"],
        output_root=str(tmp_path),
        dataset_name="lightonai/embeddings-fine-tuning",
        stages=["artifacts", "positives", "negatives", "jsonl"],
    )

    assert [name for name, _ in calls] == ["artifacts", "positives", "negatives", "jsonl"] * 2
    assert calls[0][1]["output_dir"] == str(tmp_path / "fiqa")
    assert calls[4][1]["output_dir"] == str(tmp_path / "nq")
    assert calls[1][1]["queries_path"] == str(tmp_path / "fiqa" / "queries.parquet")
    assert calls[1][1]["output_path"] == str(tmp_path / "fiqa" / "relevant_with_score.parquet")
    assert calls[2][1]["num_negatives"] == 7
    assert calls[2][1]["max_budget"] == 0
    assert calls[2][1]["candidate_score_column"] == "candidate_ranking"
    assert calls[3][1]["output_path"] == str(tmp_path / "fiqa" / "train.jsonl")
    assert calls[3][1]["positive_original_score_column"] == "lightonai_positive_score"
    assert calls[3][1]["negative_original_score_column"] == "candidate_ranking"


def test_score_flag_embedding_jsonl_moves_existing_scores_and_writes_reranker_scores(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_rows = [
        {
            "query": "query one",
            "pos": ["positive"],
            "neg": ["negative a", "negative b"],
            "pos_scores": [0.1],
            "neg_scores": [0.2, 0.3],
            "prompt": "",
            "type": "retrieval",
        },
        {
            "query": "query two",
            "pos": ["other positive"],
            "neg": ["other negative"],
            "prompt": "",
            "type": "retrieval",
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in input_rows) + "\n", encoding="utf-8")

    import curated_negatives.score_flag_embedding_jsonl as scorer

    monkeypatch.setattr(scorer, "get_reranker_model", lambda _model_name: (object(), object()))

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size, model_name):
        assert batch_size == 2
        assert model_name == "final-reranker"
        return [float(len(query) + len(doc)) for query, doc in zip(queries, docs)]

    monkeypatch.setattr(scorer, "rerank", fake_rerank)

    report = score_flag_embedding_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=2,
        record_batch_size=1,
        resume=True,
    )

    rows = _jsonl_rows(output_path)
    assert report["newly_scored_rows"] == 2
    assert rows[0]["original_pos_scores"] == [0.1]
    assert rows[0]["original_neg_scores"] == [0.2, 0.3]
    assert rows[0]["pos_scores"] == [len("query one") + len("positive")]
    assert rows[0]["neg_scores"] == [
        len("query one") + len("negative a"),
        len("query one") + len("negative b"),
    ]
    assert "original_pos_scores" not in rows[1]
    assert rows[1]["pos_scores"] == [len("query two") + len("other positive")]
    assert rows[1]["neg_scores"] == [len("query two") + len("other negative")]


def test_score_flag_embedding_jsonl_resume_skips_rows_from_incomplete_file(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_rows = [
        {"query": "q1", "pos": ["p1"], "neg": ["n1"], "prompt": "", "type": "retrieval"},
        {"query": "q2", "pos": ["p2"], "neg": ["n2"], "prompt": "", "type": "retrieval"},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in input_rows) + "\n", encoding="utf-8")

    first_scored = {
        "query": "q1",
        "pos": ["p1"],
        "neg": ["n1"],
        "prompt": "",
        "type": "retrieval",
        "pos_scores": [11.0],
        "neg_scores": [12.0],
    }
    Path(incomplete_output_path(str(output_path))).write_text(json.dumps(first_scored) + "\n", encoding="utf-8")

    import curated_negatives.score_flag_embedding_jsonl as scorer

    monkeypatch.setattr(scorer, "get_reranker_model", lambda _model_name: (object(), object()))
    calls = []

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size, model_name):
        calls.append((list(queries), list(docs), batch_size, model_name))
        return [21.0, 22.0]

    monkeypatch.setattr(scorer, "rerank", fake_rerank)

    report = score_flag_embedding_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=4,
        record_batch_size=10,
        resume=True,
    )

    rows = _jsonl_rows(output_path)
    assert report["resumed_rows"] == 1
    assert report["newly_scored_rows"] == 1
    assert rows[0] == first_scored
    assert rows[1]["query"] == "q2"
    assert rows[1]["pos_scores"] == [21.0]
    assert rows[1]["neg_scores"] == [22.0]
    assert calls == [(["q2", "q2"], ["p2", "n2"], 4, "final-reranker")]
