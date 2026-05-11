import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from find_negatives import (
    OOMRetryReranker,
    get_negative_mining_stage_defaults,
    parse_batch_size_candidates,
    validate_batch_size_options,
)


def test_parse_batch_size_candidates_builds_powers_and_includes_max():
    assert parse_batch_size_candidates(None, minimum=3, maximum=20) == [3, 6, 12, 20]


def test_parse_batch_size_candidates_sorts_positive_unique_csv_values():
    assert parse_batch_size_candidates("16,4, 4,0,-1,8", minimum=1, maximum=32) == [4, 8, 16]


def test_negative_mining_stage_defaults_use_candidate_threshold_env(monkeypatch):
    monkeypatch.setenv("BETA", "0.2")
    monkeypatch.setenv("U_FLOOR", "0.1")
    monkeypatch.setenv("CANDIDATE_BETA", "0.01")
    monkeypatch.setenv("CANDIDATE_U_FLOOR", "0.005")
    monkeypatch.setenv("CANDIDATE_TARGET", "40")
    monkeypatch.setenv("CANDIDATE_SEARCH_CHUNK", "128")
    monkeypatch.setenv("CANDIDATE_MAX_OFFSET_ITERS", "10")
    monkeypatch.setenv("CANDIDATE_RANDOM_FALLBACK", "256")

    assert get_negative_mining_stage_defaults() == {
        "candidate_beta": 0.01,
        "candidate_u_floor": 0.005,
        "candidate_target": 40,
        "candidate_search_chunk": 128,
        "candidate_max_offset_iters": 10,
        "candidate_random_fallback": 256,
    }


def test_oom_retry_reranker_halves_batch_until_call_succeeds():
    calls = []

    def fake_rerank(_tokenizer, _model, _query, answers, batch_size):
        calls.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        return [0.5] * len(answers)

    reranker = OOMRetryReranker(fake_rerank, initial_batch_size=8)

    assert reranker(object(), object(), "query", ["a", "b", "c"], batch_size=8) == [0.5, 0.5, 0.5]
    assert calls == [8, 4, 2]
    assert reranker.current_batch_size == 2


def test_validate_batch_size_options_rejects_invalid_memory_utilization():
    with pytest.raises(ValueError, match="memory_utilization"):
        validate_batch_size_options(
            explicit_batch_size=None,
            minimum=1,
            maximum=8,
            sample_size=16,
            memory_utilization=1.1,
            option_name="reranker",
        )
