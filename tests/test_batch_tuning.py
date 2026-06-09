import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

import batch_tuning
from batch_tuning import _transient_peak_extra


def test_transient_peak_extra_excludes_persistent_model_move_to_gpu():
    gib = 1024**3

    assert _transient_peak_extra(
        baseline_allocated=0,
        peak_allocated=18 * gib,
        final_allocated=17 * gib,
    ) == gib


def test_transient_peak_extra_keeps_regular_batch_allocation():
    gib = 1024**3

    assert _transient_peak_extra(
        baseline_allocated=17 * gib,
        peak_allocated=20 * gib,
        final_allocated=17 * gib,
    ) == 3 * gib


def test_benchmark_does_not_treat_initial_model_move_as_batch_memory(monkeypatch):
    gib = 1024**3
    allocated = iter([0, 17 * gib, 17 * gib, 17 * gib])
    peaks = iter([18 * gib, 19 * gib])
    clock = iter([0.0, 1.0, 2.0, 3.0])
    calls = []

    monkeypatch.setattr(batch_tuning, "clear_cuda_cache", lambda _device_id: None)
    monkeypatch.setattr(batch_tuning, "synchronize_cuda", lambda _device_id: None)
    monkeypatch.setattr(batch_tuning, "_reset_cuda_peak_memory", lambda _device_id: None)
    monkeypatch.setattr(batch_tuning, "_cuda_memory_allocated", lambda _device_id: next(allocated))
    monkeypatch.setattr(batch_tuning, "_cuda_peak_memory_allocated", lambda _device_id: next(peaks))
    monkeypatch.setattr(batch_tuning, "_cuda_free_memory", lambda _device_id: 20 * gib)
    monkeypatch.setattr(batch_tuning.time, "perf_counter", lambda: next(clock))

    selected = batch_tuning.benchmark_batch_size(
        label="reranker",
        item_label="pairs",
        sample_count=2,
        candidates=[1, 2],
        run_once=lambda candidate: calls.append(candidate),
        device_id=0,
    )

    assert calls == [1, 2]
    assert selected == 1


def test_reranker_benchmark_warms_up_one_pair(monkeypatch):
    rerank_calls = []
    benchmark_calls = []

    def fake_rerank(_tokenizer, _model, queries, docs, batch_size):
        rerank_calls.append((queries, docs, batch_size))

    def fake_benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        return 2

    monkeypatch.setattr(batch_tuning, "synchronize_cuda", lambda _device_id: None)
    monkeypatch.setattr(batch_tuning, "clear_cuda_cache", lambda _device_id: None)
    monkeypatch.setattr(batch_tuning, "benchmark_batch_size", fake_benchmark)

    selected = batch_tuning.benchmark_reranker_batch_size(
        reranker_tokenizer=object(),
        reranker_model=object(),
        sample_queries=["q1", "q2"],
        sample_docs=["d1", "d2"],
        candidates=[1, 2],
        rerank_function=fake_rerank,
    )

    assert selected == 2
    assert rerank_calls == [(["q1"], ["d1"], 1)]
    assert benchmark_calls[0]["sample_count"] == 2
