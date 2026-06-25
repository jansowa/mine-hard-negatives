import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from create_flag_embedding_jsonl import is_near_duplicate, process_negatives_streaming, token_overlap_ratio


def test_token_overlap_uses_word_counts_and_shorter_text_denominator():
    assert token_overlap_ratio("Alpha, beta gamma delta epsilon", "delta gamma beta alpha zeta") == 0.8
    assert token_overlap_ratio("one one two three", "three two one one") == 1.0
    assert token_overlap_ratio("one one one two", "one two two two") == 0.5
    assert is_near_duplicate("delta gamma beta alpha zeta", ["Alpha, beta gamma delta epsilon"], 0.8)


def test_process_negatives_streaming_reads_custom_negative_score_column(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": ["p1", "n1"],
            "text": ["positive text", "negative text"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame({"id": ["q1"], "text": ["question"]}).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["p1"],
            "positive_ranking": [1.0],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["n1"],
            "final_ranking": [0.5],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=1,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=1,
        oversample_factor=1,
        low_memory_optimizations=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["neg_id"] == ["n1"]
    assert rows[0]["neg_scores"] == [0.5]
    assert rows[0]["pos_is_synthetic"] == [False]


def test_process_negatives_streaming_backfills_relaxed_candidates(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": ["p1", "n1", "n2"],
            "text": ["positive text", "strict negative", "relaxed negative"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame({"id": ["q1"], "text": ["question"]}).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["p1"],
            "positive_ranking": [1.0],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "document_id": ["n1", "n2", "p1"],
            "final_ranking": [0.5, 2.0, 0.1],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=2,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=1,
        oversample_factor=1,
        backfill_policy="relaxed",
        report_path=str(tmp_path / "report.json"),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["neg_id"] == ["n1", "n2"]
    assert rows[0]["neg_selection_tier"] == ["strict", "relaxed_backfill"]
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["low_memory_optimizations"] is False


def test_process_negatives_streaming_exports_prompt_type_and_original_scores(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": ["p1", "n1", "n2"],
            "text": ["positive text", "strict negative", "relaxed negative"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame({"id": ["q1"], "text": ["question"]}).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["p1"],
            "positive_ranking": [1.0],
            "lightonai_positive_score": [0.95],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "document_id": ["n1", "n2"],
            "final_ranking": [0.5, 2.0],
            "candidate_ranking": [0.4, 0.2],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=2,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        positive_original_score_column="lightonai_positive_score",
        negative_original_score_column="candidate_ranking",
        prompt="Represent this sentence for searching relevant passages:",
        dataset_type="retrieval",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=1,
        oversample_factor=1,
        backfill_policy="relaxed",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prompt"] == "Represent this sentence for searching relevant passages:"
    assert rows[0]["type"] == "retrieval"
    assert rows[0]["pos_scores"] == [1.0]
    assert rows[0]["neg_scores"] == [0.5, 2.0]
    assert rows[0]["original_pos_scores"] == [0.95]
    assert rows[0]["original_neg_scores"] == [0.4, 0.2]
    assert rows[0]["neg_selection_tier"] == ["strict", "relaxed_backfill"]
    assert rows[0]["pos_is_synthetic"] == [False]


def test_process_negatives_streaming_ignores_missing_optional_original_score_columns(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": ["p1", "n1"],
            "text": ["positive text", "negative text"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame({"id": ["q1"], "text": ["question"]}).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["p1"],
            "qrel_score": [1],
            "qrel_split": ["train"],
            "positive_ranking": [1.0],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["n1"],
            "final_ranking": [0.5],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=1,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        positive_original_score_column="lightonai_positive_score",
        negative_original_score_column="candidate_ranking",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=1,
        oversample_factor=1,
        backfill_policy="relaxed",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["pos_id"] == ["p1"]
    assert rows[0]["neg_id"] == ["n1"]
    assert "original_pos_scores" not in rows[0]
    assert "original_neg_scores" not in rows[0]


def test_process_negatives_streaming_mines_positives_from_existing_scores(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": [
                "p1",
                "p2",
                "p3",
                "p4",
                "absolute",
                "synthetic_duplicate",
                "duplicate",
                "beta",
                "negative",
            ],
            "text": [
                "alpha beta gamma delta epsilon",
                "positive two",
                "positive three",
                "positive four",
                "an independently excellent answer",
                "answer excellent independently an extra",
                "epsilon delta gamma beta alpha",
                "another strong and useful answer",
                "irrelevant passage",
            ],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame(
        {
            "id": ["q1", "q2", "q3", "q4"],
            "text": ["question one", "question two", "question three", "question four"],
        }
    ).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3", "q4"],
            "document_id": ["p1", "p2", "p3", "p4"],
            "positive_ranking": [40.0, 10.0, 30.0, 20.0],
            "original_positive": [0.9, 0.8, 0.7, 0.6],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q1", "q1"],
            "document_id": ["absolute", "synthetic_duplicate", "duplicate", "beta", "negative"],
            "final_ranking": [45.0, 42.0, 39.0, 35.0, 1.0],
            "candidate_ranking": [0.95, 0.945, 0.94, 0.85, 0.1],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=1,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        positive_original_score_column="original_positive",
        negative_original_score_column="candidate_ranking",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=4,
        oversample_factor=1,
        mine_positives=True,
        max_mined_positives=2,
        u_sanity_ceiling=0.75,
        u_absolute_ceiling=0.99,
        u_positive_beta=0.80,
        positive_near_duplicate_threshold=0.80,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["pos_id"] == ["p1", "absolute", "beta"]
    assert rows[0]["pos_scores"] == [40.0, 45.0, 35.0]
    assert rows[0]["original_pos_scores"] == [0.9, 0.95, 0.85]
    assert rows[0]["pos_is_synthetic"] == [False, True, True]
    assert rows[0]["neg_id"] == ["negative"]
