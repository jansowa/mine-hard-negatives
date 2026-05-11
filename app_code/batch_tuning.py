import logging
import threading
import time
from collections.abc import Callable, Sequence


def _import_torch():
    import torch

    return torch


def _import_torch_or_none():
    try:
        return _import_torch()
    except ImportError:
        return None


def _info(logger: logging.Logger | None, message: str, *args) -> None:
    if logger is not None:
        logger.info(message, *args)
    elif args:
        print(message % args)
    else:
        print(message)


def _warning(logger: logging.Logger | None, message: str, *args) -> None:
    if logger is not None:
        logger.warning(message, *args)
    elif args:
        print(("WARNING: " + message) % args)
    else:
        print(f"WARNING: {message}")


def parse_batch_size_candidates(raw_candidates: str | None, minimum: int, maximum: int) -> list[int]:
    if raw_candidates:
        candidates = [int(item.strip()) for item in raw_candidates.split(",") if item.strip()]
    else:
        candidates = []
        value = minimum
        while value <= maximum:
            candidates.append(value)
            value *= 2
        if maximum not in candidates:
            candidates.append(maximum)

    candidates = sorted({candidate for candidate in candidates if candidate > 0})
    if not candidates:
        raise ValueError("Auto batch-size candidates must contain at least one positive integer")
    return candidates


def validate_batch_size_options(
    explicit_batch_size: int | None,
    minimum: int,
    maximum: int,
    sample_size: int,
    memory_utilization: float,
    option_name: str,
) -> None:
    if explicit_batch_size is not None and explicit_batch_size <= 0:
        raise ValueError(f"--{option_name}_batch_size must be greater than 0")
    if minimum <= 0 or maximum <= 0:
        raise ValueError(
            f"--auto_{option_name}_batch_size_min and --auto_{option_name}_batch_size_max must be greater than 0"
        )
    if minimum > maximum:
        raise ValueError(
            f"--auto_{option_name}_batch_size_min cannot be greater than --auto_{option_name}_batch_size_max"
        )
    if sample_size < 0:
        raise ValueError(f"--auto_{option_name}_batch_size_sample_size must be greater than or equal to 0")
    if not 0 < memory_utilization <= 1:
        raise ValueError(f"--auto_{option_name}_batch_size_memory_utilization must be in the (0, 1] range")


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def synchronize_cuda(device_id: int | None = None) -> None:
    try:
        torch = _import_torch_or_none()
        if torch is None:
            return
        if torch.cuda.is_available():
            if device_id is None:
                for gpu_id in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(gpu_id)
            else:
                torch.cuda.synchronize(device_id)
    except Exception:
        return


def clear_cuda_cache(device_id: int | None = None) -> None:
    try:
        torch = _import_torch_or_none()
        if torch is None:
            return
        if torch.cuda.is_available():
            if device_id is None:
                for gpu_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            else:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            synchronize_cuda(device_id)
    except Exception:
        return


def set_dense_embedding_batch_size(dense_embeddings, batch_size: int) -> None:
    encode_kwargs = getattr(dense_embeddings, "encode_kwargs", None)
    if isinstance(encode_kwargs, dict):
        encode_kwargs["batch_size"] = batch_size
    if hasattr(dense_embeddings, "batch_size"):
        dense_embeddings.batch_size = batch_size


def embed_dense_documents(dense_embeddings, texts: list[str]):
    client = getattr(dense_embeddings, "_client", None)
    encode_kwargs = getattr(dense_embeddings, "encode_kwargs", None)
    if client is not None and isinstance(encode_kwargs, dict) and hasattr(client, "encode"):
        texts = [text.replace("\n", " ") for text in texts]
        return client.encode(
            texts,
            show_progress_bar=getattr(dense_embeddings, "show_progress", False),
            **encode_kwargs,
        )

    return dense_embeddings.embed_documents(texts)


def _cuda_memory_allocated(device_id: int | None) -> int | None:
    torch = _import_torch_or_none()
    if torch is None:
        return None
    if device_id is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.memory_allocated(device_id)
    except Exception:
        return None


def _cuda_peak_memory_allocated(device_id: int | None) -> int | None:
    torch = _import_torch_or_none()
    if torch is None:
        return None
    if device_id is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated(device_id)
    except Exception:
        return None


def _reset_cuda_peak_memory(device_id: int | None) -> None:
    torch = _import_torch_or_none()
    if torch is None:
        return
    if device_id is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device_id)
    except Exception:
        return


def _cuda_free_memory(device_id: int | None) -> int | None:
    torch = _import_torch_or_none()
    if torch is None:
        return None
    if device_id is None or not torch.cuda.is_available():
        return None
    try:
        free, _ = torch.cuda.mem_get_info(device_id)
        return free
    except Exception:
        return None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"

    out = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if out < 1024 or unit == "GiB":
            return f"{out:.1f} {unit}"
        out /= 1024
    return f"{out:.1f} GiB"


def _should_skip_candidate_for_memory(
    candidate: int,
    previous_candidate: int | None,
    previous_peak_extra: int | None,
    device_id: int | None,
    memory_utilization: float,
) -> bool:
    if previous_candidate is None or previous_peak_extra is None or previous_peak_extra <= 0:
        return False

    free_memory = _cuda_free_memory(device_id)
    if free_memory is None:
        return False

    estimated_extra = int(previous_peak_extra * (candidate / previous_candidate))
    allowed_extra = int(free_memory * memory_utilization)
    return estimated_extra > allowed_extra


def benchmark_batch_size(
    label: str,
    item_label: str,
    sample_count: int,
    candidates: list[int],
    run_once: Callable[[int], None],
    device_id: int | None,
    memory_utilization: float = 0.70,
    logger: logging.Logger | None = None,
) -> int:
    if sample_count <= 0:
        selected = candidates[0]
        _warning(logger, "No samples available for %s auto-tuning; using batch_size=%s", label, selected)
        return selected

    _info(logger, "Auto-tuning %s batch size on %s sample %s", label, sample_count, item_label)
    results: list[tuple[float, int]] = []
    previous_candidate = None
    previous_peak_extra = None

    for candidate in candidates:
        clear_cuda_cache(device_id)
        if _should_skip_candidate_for_memory(
            candidate,
            previous_candidate,
            previous_peak_extra,
            device_id,
            memory_utilization,
        ):
            _info(
                logger,
                "  batch_size=%s: skipped to keep CUDA memory headroom after batch_size=%s",
                candidate,
                previous_candidate,
            )
            break

        baseline_allocated = _cuda_memory_allocated(device_id)
        _reset_cuda_peak_memory(device_id)
        started_at = time.perf_counter()
        try:
            run_once(candidate)
            synchronize_cuda(device_id)
        except Exception as exc:
            clear_cuda_cache(device_id)
            if is_cuda_oom(exc):
                _info(logger, "  batch_size=%s: CUDA OOM, stopping candidate search", candidate)
                break
            _warning(logger, "  batch_size=%s: failed (%s: %s)", candidate, type(exc).__name__, exc)
            continue

        elapsed = max(time.perf_counter() - started_at, 1e-9)
        items_per_second = sample_count / elapsed
        peak_allocated = _cuda_peak_memory_allocated(device_id)
        peak_extra = None
        if baseline_allocated is not None and peak_allocated is not None:
            peak_extra = max(0, peak_allocated - baseline_allocated)

        _info(
            logger,
            "  batch_size=%s: %.1f %s/s (%.2fs, peak extra %s)",
            candidate,
            items_per_second,
            item_label,
            elapsed,
            _format_bytes(peak_extra),
        )
        results.append((items_per_second, candidate))
        previous_candidate = candidate
        previous_peak_extra = peak_extra

    if not results:
        raise RuntimeError(f"Could not find a working {label} batch size")

    best_speed = max(speed for speed, _ in results)
    selected = min(candidate for speed, candidate in results if speed >= best_speed * 0.98)
    _info(logger, "Selected %s batch size: %s", label, selected)
    return selected


def benchmark_embedding_batch_size(
    dense_embeddings,
    sample_texts: list[str],
    candidates: list[int],
    memory_utilization: float = 0.70,
    logger: logging.Logger | None = None,
) -> int:
    def run_once(candidate: int) -> None:
        set_dense_embedding_batch_size(dense_embeddings, candidate)
        embed_dense_documents(dense_embeddings, sample_texts)

    selected = benchmark_batch_size(
        label="embedding",
        item_label="texts",
        sample_count=len(sample_texts),
        candidates=candidates,
        run_once=run_once,
        device_id=0,
        memory_utilization=memory_utilization,
        logger=logger,
    )
    set_dense_embedding_batch_size(dense_embeddings, selected)
    return selected


def benchmark_reranker_batch_size(
    reranker_tokenizer,
    reranker_model,
    sample_queries: Sequence[str],
    sample_docs: Sequence[str],
    candidates: list[int],
    rerank_function: Callable,
    memory_utilization: float = 0.70,
    device_id: int | None = 0,
    label: str = "reranker",
    logger: logging.Logger | None = None,
) -> int:
    queries = list(sample_queries)
    docs = list(sample_docs)

    def run_once(candidate: int) -> None:
        rerank_function(
            reranker_tokenizer,
            reranker_model,
            queries,
            docs,
            batch_size=candidate,
        )

    return benchmark_batch_size(
        label=label,
        item_label="pairs",
        sample_count=len(docs),
        candidates=candidates,
        run_once=run_once,
        device_id=device_id,
        memory_utilization=memory_utilization,
        logger=logger,
    )


class OOMRetryReranker:
    def __init__(self, rerank_function: Callable, initial_batch_size: int, logger: logging.Logger | None = None):
        self.rerank_function = rerank_function
        self.current_batch_size = max(1, initial_batch_size)
        self.logger = logger
        self.lock = threading.Lock()

    def __call__(self, tokenizer, model, query, answers: list[str], batch_size: int = 16) -> list[float]:
        requested_batch_size = max(1, int(batch_size or self.current_batch_size))
        with self.lock:
            effective_batch_size = min(requested_batch_size, self.current_batch_size)

        while True:
            try:
                return self.rerank_function(
                    tokenizer,
                    model,
                    query,
                    answers,
                    batch_size=effective_batch_size,
                )
            except Exception as exc:
                if not is_cuda_oom(exc) or effective_batch_size <= 1:
                    raise

                next_batch_size = max(1, effective_batch_size // 2)
                with self.lock:
                    self.current_batch_size = min(self.current_batch_size, next_batch_size)
                clear_cuda_cache()
                _warning(
                    self.logger,
                    "Reranker CUDA OOM at batch_size=%s; retrying with batch_size=%s",
                    effective_batch_size,
                    next_batch_size,
                )
                effective_batch_size = next_batch_size
