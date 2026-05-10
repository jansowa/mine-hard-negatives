import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

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
