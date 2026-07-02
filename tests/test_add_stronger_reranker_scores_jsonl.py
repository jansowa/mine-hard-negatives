import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from add_stronger_reranker_scores_jsonl import (
    add_stronger_reranker_scores_jsonl,
    incomplete_output_path,
)


def _jsonl_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _patch_reranker(monkeypatch, scores):
    import add_stronger_reranker_scores_jsonl as scorer

    calls = []
    monkeypatch.setattr(scorer, "get_reranker_model", lambda _model_name: (object(), object()))

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size, model_name):
        calls.append((list(queries), list(docs), batch_size, model_name))
        if callable(scores):
            return scores(queries, docs)
        return scores[: len(docs)]

    monkeypatch.setattr(scorer, "rerank", fake_rerank)
    return calls


def test_adds_stronger_positive_scores_without_changing_existing_scores(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_rows = [
        {
            "query": "q1",
            "pos": ["p1", "p2"],
            "neg": ["n1"],
            "pos_scores": [0.1, 0.2],
            "neg_scores": [0.3],
        },
        {"query": "q2", "pos": ["p3"], "neg": ["n2"], "pos_scores": [0.4], "neg_scores": [0.5]},
    ]
    _write_jsonl(input_path, input_rows)
    calls = _patch_reranker(monkeypatch, lambda queries, docs: [float(len(q) + len(d)) for q, d in zip(queries, docs)])

    report = add_stronger_reranker_scores_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=2,
        record_batch_size=10,
    )

    rows = _jsonl_rows(output_path)
    assert report["newly_scored_pairs"] == 3
    assert rows[0]["pos_scores"] == [0.1, 0.2]
    assert rows[0]["neg_scores"] == [0.3]
    assert rows[0]["pos_scores_stronger_reranker"] == [4.0, 4.0]
    assert rows[1]["pos_scores"] == [0.4]
    assert rows[1]["pos_scores_stronger_reranker"] == [4.0]
    assert "neg_scores_stronger_reranker" not in rows[0]
    assert calls == [(["q1", "q1", "q2"], ["p1", "p2", "p3"], 2, "final-reranker")]


def test_score_negatives_adds_stronger_negative_scores(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_jsonl(input_path, [{"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}])
    calls = _patch_reranker(monkeypatch, [1.0, 2.0, 3.0])

    add_stronger_reranker_scores_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=4,
        record_batch_size=1,
        score_negatives=True,
    )

    rows = _jsonl_rows(output_path)
    assert rows[0]["pos_scores_stronger_reranker"] == [1.0]
    assert rows[0]["neg_scores_stronger_reranker"] == [2.0, 3.0]
    assert calls == [(["q", "q", "q"], ["p", "n1", "n2"], 4, "final-reranker")]


def test_resume_skips_complete_rows_from_incomplete_file(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_rows = [
        {"query": "q1", "pos": ["p1"], "neg": ["n1"]},
        {"query": "q2", "pos": ["p2"], "neg": ["n2"]},
    ]
    _write_jsonl(input_path, input_rows)
    first_scored = {"query": "q1", "pos": ["p1"], "neg": ["n1"], "pos_scores_stronger_reranker": [11.0]}
    Path(incomplete_output_path(str(output_path))).write_text(json.dumps(first_scored) + "\n", encoding="utf-8")
    calls = _patch_reranker(monkeypatch, [22.0])

    report = add_stronger_reranker_scores_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=2,
        record_batch_size=10,
        resume=True,
    )

    rows = _jsonl_rows(output_path)
    assert report["resumed_rows"] == 1
    assert report["newly_scored_rows"] == 1
    assert rows[0] == first_scored
    assert rows[1]["pos_scores_stronger_reranker"] == [22.0]
    assert calls == [(["q2"], ["p2"], 2, "final-reranker")]


def test_only_missing_pairs_are_scored(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "query": "q",
                "pos": ["p1", "p2", "p3"],
                "neg": ["n1", "n2"],
                "pos_scores_stronger_reranker": [10.0, None],
                "neg_scores_stronger_reranker": [None, 99.0],
            }
        ],
    )
    calls = _patch_reranker(monkeypatch, [20.0, 30.0, 40.0])

    report = add_stronger_reranker_scores_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=8,
        record_batch_size=1,
        score_negatives=True,
    )

    rows = _jsonl_rows(output_path)
    assert report["newly_scored_pairs"] == 3
    assert rows[0]["pos_scores_stronger_reranker"] == [10.0, 20.0, 30.0]
    assert rows[0]["neg_scores_stronger_reranker"] == [40.0, 99.0]
    assert calls == [(["q", "q", "q"], ["p2", "p3", "n1"], 8, "final-reranker")]


def test_rejects_score_lists_longer_than_text_lists(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_jsonl(input_path, [{"query": "q", "pos": ["p"], "pos_scores_stronger_reranker": [1.0, 2.0]}])
    _patch_reranker(monkeypatch, [])

    with pytest.raises(ValueError, match="pos_scores_stronger_reranker has 2 values"):
        add_stronger_reranker_scores_jsonl(
            input_path=str(input_path),
            output_path=str(output_path),
            reranker_model_name="final-reranker",
            reranker_batch_size=1,
            record_batch_size=1,
        )


def test_ignores_null_values_in_original_score_fields(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_jsonl(input_path, [{"query": "q", "pos": ["p"], "neg": [], "pos_scores": [None]}])
    _patch_reranker(monkeypatch, [5.0])

    add_stronger_reranker_scores_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        reranker_model_name="final-reranker",
        reranker_batch_size=1,
        record_batch_size=1,
    )

    rows = _jsonl_rows(output_path)
    assert rows[0]["pos_scores"] == [None]
    assert rows[0]["pos_scores_stronger_reranker"] == [5.0]
