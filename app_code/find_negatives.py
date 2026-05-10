import argparse
import logging
import os
import threading
import time
from datetime import datetime
from functools import partial

import pandas as pd
import torch
from decouple import config
from tqdm import tqdm

from config import get_vector_db_backend
from models import get_dense_model, get_sparse_model, rerank
from multi_gpu_processor import (
    MultiGPUNegativeFinder,
    setup_multi_gpu_models,
    should_resume_processing,
)
from utils.vector_db import create_vector_backend

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or exc.__class__.__name__ == "OutOfMemoryError"


def clear_cuda_cache(device_id: int | None = None) -> None:
    try:
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


def synchronize_cuda(device_id: int | None = None) -> None:
    try:
        if torch.cuda.is_available():
            if device_id is None:
                for gpu_id in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(gpu_id)
            else:
                torch.cuda.synchronize(device_id)
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


def collect_query_text_sample(queries: list[dict], sample_size: int) -> list[str]:
    if sample_size <= 0:
        return []

    sample: list[str] = []
    for query in queries:
        text = query.get("text")
        if text:
            sample.append(str(text))
            if len(sample) >= sample_size:
                break
    return sample


def _cuda_memory_allocated(device_id: int | None) -> int | None:
    if device_id is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.memory_allocated(device_id)
    except Exception:
        return None


def _cuda_peak_memory_allocated(device_id: int | None) -> int | None:
    if device_id is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.max_memory_allocated(device_id)
    except Exception:
        return None


def _reset_cuda_peak_memory(device_id: int | None) -> None:
    if device_id is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device_id)
    except Exception:
        return


def _cuda_free_memory(device_id: int | None) -> int | None:
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
    run_once,
    device_id: int | None,
    memory_utilization: float,
) -> int:
    if sample_count <= 0:
        selected = candidates[0]
        logger.warning("No samples available for %s auto-tuning; using batch_size=%s", label, selected)
        return selected

    logger.info("Auto-tuning %s batch size on %s sample %s", label, sample_count, item_label)
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
            logger.info(
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
                logger.info("  batch_size=%s: CUDA OOM, stopping candidate search", candidate)
                break
            logger.warning("  batch_size=%s: failed (%s: %s)", candidate, type(exc).__name__, exc)
            continue

        elapsed = max(time.perf_counter() - started_at, 1e-9)
        items_per_second = sample_count / elapsed
        peak_allocated = _cuda_peak_memory_allocated(device_id)
        peak_extra = None
        if baseline_allocated is not None and peak_allocated is not None:
            peak_extra = max(0, peak_allocated - baseline_allocated)

        logger.info(
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
    logger.info("Selected %s batch size: %s", label, selected)
    return selected


def benchmark_embedding_batch_size(
    dense_embeddings,
    sample_texts: list[str],
    candidates: list[int],
    memory_utilization: float,
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
    )
    set_dense_embedding_batch_size(dense_embeddings, selected)
    return selected


def collect_reranker_sample_pairs(
    queries: list[dict],
    vector_backend,
    sample_size: int,
) -> tuple[list[str], list[str]]:
    if sample_size <= 0 or not queries:
        return [], []

    sample_queries: list[str] = []
    sample_docs: list[str] = []
    docs_per_query = min(8, sample_size)

    for query in queries:
        remaining = sample_size - len(sample_docs)
        if remaining <= 0:
            break

        qtext = str(query.get("text") or "")
        if not qtext:
            continue

        try:
            docs = vector_backend.search(qtext, k=min(docs_per_query, remaining), offset=0)
        except Exception as exc:
            logger.warning("Could not collect search-based reranker tuning samples (%s: %s)", type(exc).__name__, exc)
            break

        for doc in docs:
            doc_text = getattr(doc, "page_content", "")
            if not doc_text:
                continue
            sample_queries.append(qtext)
            sample_docs.append(str(doc_text))
            if len(sample_docs) >= sample_size:
                break

    if not sample_docs:
        logger.warning("Using query texts as fallback reranker tuning samples")
        for query in queries[:sample_size]:
            qtext = str(query.get("text") or "")
            if qtext:
                sample_queries.append(qtext)
                sample_docs.append(qtext)

    return sample_queries, sample_docs


def benchmark_reranker_batch_size(
    model_sets,
    rerank_function,
    sample_queries: list[str],
    sample_docs: list[str],
    candidates: list[int],
    memory_utilization: float,
) -> int:
    selected_by_gpu: list[tuple[int, int]] = []
    seen_gpu_ids: set[int] = set()

    for model_set in model_sets:
        gpu_id = int(getattr(model_set, "gpu_id", 0))
        if gpu_id in seen_gpu_ids:
            continue
        seen_gpu_ids.add(gpu_id)

        def run_once(candidate: int, current_model_set=model_set) -> None:
            rerank_function(
                current_model_set.reranker_tokenizer,
                current_model_set.reranker_model,
                sample_queries,
                sample_docs,
                batch_size=candidate,
            )

        selected = benchmark_batch_size(
            label=f"reranker GPU {gpu_id}",
            item_label="pairs",
            sample_count=len(sample_docs),
            candidates=candidates,
            run_once=run_once,
            device_id=gpu_id,
            memory_utilization=memory_utilization,
        )
        selected_by_gpu.append((gpu_id, selected))

    if not selected_by_gpu:
        return candidates[0]

    selected = min(batch_size for _, batch_size in selected_by_gpu)
    if len(selected_by_gpu) > 1:
        details = ", ".join(f"gpu{gpu_id}={batch_size}" for gpu_id, batch_size in selected_by_gpu)
        logger.info("Selected global reranker batch size: %s (%s)", selected, details)
    return selected


class OOMRetryReranker:
    def __init__(self, rerank_function, initial_batch_size: int):
        self.rerank_function = rerank_function
        self.current_batch_size = max(1, initial_batch_size)
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
                logger.warning(
                    "Reranker CUDA OOM at batch_size=%s; retrying with batch_size=%s",
                    effective_batch_size,
                    next_batch_size,
                )
                effective_batch_size = next_batch_size


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


def find_negatives_multigpu(
    dense_model_name: str,
    sparse_model_name: str,
    embedding_batch_size: int | None,
    reranker_model_name: str,
    reranker_batch_size: int | None,
    collection_name: str,
    queries_path: str,
    relevant_path: str,
    output_path: str,
    top_k: int,
    force_resume: bool | None = None,
    query_batch_size: int = config("NEGATIVE_QUERY_BATCH_SIZE", cast=int, default=4),
    profile_timing: bool = config("NEGATIVE_PROFILE_TIMING", cast=bool, default=False),
    auto_embedding_batch_size_candidates: str | None = config(
        "NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_CANDIDATES", default=None
    ),
    auto_embedding_batch_size_min: int = config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MIN", cast=int, default=1),
    auto_embedding_batch_size_max: int = config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MAX", cast=int, default=64),
    auto_embedding_batch_size_sample_size: int = config(
        "NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=64
    ),
    auto_embedding_batch_size_memory_utilization: float = config(
        "NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MEMORY_UTILIZATION",
        cast=float,
        default=0.70,
    ),
    auto_reranker_batch_size_candidates: str | None = config(
        "NEGATIVE_AUTO_RERANKER_BATCH_SIZE_CANDIDATES", default=None
    ),
    auto_reranker_batch_size_min: int = config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MIN", cast=int, default=1),
    auto_reranker_batch_size_max: int = config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MAX", cast=int, default=64),
    auto_reranker_batch_size_sample_size: int = config(
        "NEGATIVE_AUTO_RERANKER_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=128
    ),
    auto_reranker_batch_size_memory_utilization: float = config(
        "NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MEMORY_UTILIZATION",
        cast=float,
        default=0.70,
    ),
    ranking_column: str = config("NEGATIVE_RANKING_COLUMN", default="ranking"),
    positive_score_column: str = config("NEGATIVE_POSITIVE_SCORE_COLUMN", default="positive_ranking"),
):
    validate_batch_size_options(
        embedding_batch_size,
        auto_embedding_batch_size_min,
        auto_embedding_batch_size_max,
        auto_embedding_batch_size_sample_size,
        auto_embedding_batch_size_memory_utilization,
        "embedding",
    )
    validate_batch_size_options(
        reranker_batch_size,
        auto_reranker_batch_size_min,
        auto_reranker_batch_size_max,
        auto_reranker_batch_size_sample_size,
        auto_reranker_batch_size_memory_utilization,
        "reranker",
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    if not ranking_column:
        raise ValueError("--ranking_column must not be empty")
    if not positive_score_column:
        raise ValueError("--positive_score_column must not be empty")

    logger.info(f"{num_gpus} GPUs available for processing")
    logger.info("Loading queries and relevance data")
    queries_df = pd.read_parquet(queries_path)
    relevant_df = pd.read_parquet(relevant_path)

    queries_df["id"] = queries_df["id"].astype("string")
    relevant_df["query_id"] = relevant_df["query_id"].astype("string")
    relevant_df["document_id"] = relevant_df["document_id"].astype("string")

    if positive_score_column not in relevant_df.columns:
        raise ValueError(f"Positive score column {positive_score_column!r} is missing from {relevant_path}")

    best_relevant_df = relevant_df.loc[relevant_df.groupby("query_id")[positive_score_column].idxmax()]
    positives_df = queries_df.merge(best_relevant_df, left_on="id", right_on="query_id").drop(columns="id")

    queries_list = [
        {
            "query_id": row["query_id"],
            "document_id": row["document_id"],
            "positive_score": row[positive_score_column],
            "text": row["text"],
        }
        for _, row in positives_df.iterrows()
    ]

    logger.info(f"Loaded {len(queries_list)} queries for processing")

    effective_embedding_batch_size = embedding_batch_size or auto_embedding_batch_size_min
    dense_model_vs = get_dense_model(
        dense_model_name,
        batch_size=effective_embedding_batch_size,
        prompt=config("DENSE_PROMPT"),
        gpu_id=0,
    )
    if embedding_batch_size is None:
        embedding_candidates = parse_batch_size_candidates(
            auto_embedding_batch_size_candidates,
            auto_embedding_batch_size_min,
            auto_embedding_batch_size_max,
        )
        embedding_sample = collect_query_text_sample(queries_list, auto_embedding_batch_size_sample_size)
        effective_embedding_batch_size = benchmark_embedding_batch_size(
            dense_model_vs,
            embedding_sample,
            embedding_candidates,
            auto_embedding_batch_size_memory_utilization,
        )
    else:
        set_dense_embedding_batch_size(dense_model_vs, effective_embedding_batch_size)
        logger.info("Using explicit embedding batch size: %s", effective_embedding_batch_size)

    backend_type = get_vector_db_backend()
    sparse_model_vs = None
    dense_dim_size = None
    if backend_type == "qdrant":
        sparse_model_vs = get_sparse_model(sparse_model_name, batch_size=effective_embedding_batch_size, gpu_id=0)
        dense_dim_size = len(dense_model_vs.embed_query("text"))

    model_sets = setup_multi_gpu_models(
        reranker_model_name,
        relevant_path,
        models_per_gpu=1,
        logger=logger,
        positive_score_column=positive_score_column,
    )

    rerank_function = partial(rerank, model_name=reranker_model_name)

    if force_resume is not None:
        resume = force_resume
        logger.info(f"Resume mode forced to: {resume}")
    else:
        resume = should_resume_processing(output_path, model_sets)

    logger.info("Detected existing worker files - resuming processing" if resume else "Starting fresh processing")

    logger.info("Setting up vector backend")
    vector_backend = create_vector_backend(
        collection_name=collection_name,
        dense_embeddings=dense_model_vs,
        sparse_embeddings=sparse_model_vs,
        dense_dim_size=dense_dim_size,
    )

    if reranker_batch_size is None:
        reranker_candidates = parse_batch_size_candidates(
            auto_reranker_batch_size_candidates,
            auto_reranker_batch_size_min,
            auto_reranker_batch_size_max,
        )
        sample_queries, sample_docs = collect_reranker_sample_pairs(
            queries_list,
            vector_backend,
            auto_reranker_batch_size_sample_size,
        )
        reranker_batch_size = benchmark_reranker_batch_size(
            model_sets,
            rerank_function,
            sample_queries,
            sample_docs,
            reranker_candidates,
            auto_reranker_batch_size_memory_utilization,
        )
    else:
        logger.info("Using explicit reranker batch size: %s", reranker_batch_size)

    safe_rerank_function = OOMRetryReranker(rerank_function, reranker_batch_size)
    logger.info("Effective embedding batch size: %s", effective_embedding_batch_size)
    logger.info("Effective reranker batch size: %s", reranker_batch_size)

    processor = MultiGPUNegativeFinder(
        model_sets,
        output_path=output_path,
        progress_bar=None,
        logger=logger,
        resume=resume,
        profile_timing=profile_timing,
        ranking_column=ranking_column,
    )

    if resume:
        processed_query_ids = processor.get_processed_query_ids()
        remaining_queries = len([q for q in queries_list if q["query_id"] not in processed_query_ids])
        logger.info(f"Resuming: {len(processed_query_ids)} already processed, {remaining_queries} remaining")
        progress_bar = tqdm(total=remaining_queries, desc="Queries Processed", position=0)
    else:
        progress_bar = tqdm(total=len(queries_list), desc="Queries Processed", position=0)

    processor.progress_bar = progress_bar

    logger.info("Starting multi-GPU negative finding with individual worker files")
    try:
        completed_batches = processor.process_all(
            queries_list,
            vector_backend,
            safe_rerank_function,
            top_k,
            reranker_batch_size,
            query_batch_size,
        )
        logger.info(f"Successfully processed {completed_batches} batches")
        progress_bar.close()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user, but progress has been saved")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.info("Some progress may have been saved to the output file")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find negative samples using multi-GPU processing")
    parser.add_argument("--dense_model_name", type=str, default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, default=config("SPLADE_MODEL_NAME"))
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=None,
        help="Explicit embedder batch size. If omitted, a short startup benchmark selects it automatically.",
    )
    parser.add_argument(
        "--reranker_model_name", type=str, default=config("CANDIDATE_RERANKER_NAME", default=config("RERANKER_NAME"))
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=None,
        help="Explicit reranker batch size. If omitted, a short startup benchmark selects it automatically.",
    )
    parser.add_argument(
        "--auto_embedding_batch_size_candidates",
        type=str,
        default=config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_CANDIDATES", default=None),
    )
    parser.add_argument(
        "--auto_embedding_batch_size_min",
        type=int,
        default=config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MIN", cast=int, default=1),
    )
    parser.add_argument(
        "--auto_embedding_batch_size_max",
        type=int,
        default=config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MAX", cast=int, default=64),
    )
    parser.add_argument(
        "--auto_embedding_batch_size_sample_size",
        type=int,
        default=config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=64),
    )
    parser.add_argument(
        "--auto_embedding_batch_size_memory_utilization",
        type=float,
        default=config("NEGATIVE_AUTO_EMBEDDING_BATCH_SIZE_MEMORY_UTILIZATION", cast=float, default=0.70),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_candidates",
        type=str,
        default=config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_CANDIDATES", default=None),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_min",
        type=int,
        default=config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MIN", cast=int, default=1),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_max",
        type=int,
        default=config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MAX", cast=int, default=64),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_sample_size",
        type=int,
        default=config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=128),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_memory_utilization",
        type=float,
        default=config("NEGATIVE_AUTO_RERANKER_BATCH_SIZE_MEMORY_UTILIZATION", cast=float, default=0.70),
    )
    parser.add_argument(
        "--database_collection_name", type=str, default=config("DATABASE_COLLECTION_NAME", default="all_documents")
    )
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument(
        "--relevant_path",
        type=str,
        default=config("RELEVANT_WITH_CANDIDATE_SCORE_PATH", default=config("RELEVANT_WITH_SCORE_PATH")),
    )
    parser.add_argument(
        "--output_path", type=str, default=config("NEGATIVE_CANDIDATES_PATH", default=config("NEGATIVES_PATH"))
    )
    parser.add_argument("--top_k", type=int, default=config("TOP_K", cast=int))
    parser.add_argument(
        "--query_batch_size", type=int, default=config("NEGATIVE_QUERY_BATCH_SIZE", cast=int, default=4)
    )
    parser.add_argument(
        "--ranking_column",
        type=str,
        default=config("NEGATIVE_RANKING_COLUMN", default="ranking"),
        help="Output score column name. Use candidate_ranking for the optional first reranking stage.",
    )
    parser.add_argument(
        "--positive_score_column",
        type=str,
        default=config("NEGATIVE_POSITIVE_SCORE_COLUMN", default="positive_ranking"),
        help="Positive score column used to build the percentile threshold distribution.",
    )
    parser.add_argument("--profile-timing", dest="profile_timing", action="store_true", default=None)
    parser.add_argument("--no-profile-timing", dest="profile_timing", action="store_false", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None)

    args = parser.parse_args()

    find_negatives_multigpu(
        args.dense_model_name,
        args.sparse_model_name,
        args.embedding_batch_size,
        args.reranker_model_name,
        args.reranker_batch_size,
        args.database_collection_name,
        args.queries_path,
        args.relevant_path,
        args.output_path,
        args.top_k,
        args.resume,
        args.query_batch_size,
        config("NEGATIVE_PROFILE_TIMING", cast=bool, default=False)
        if args.profile_timing is None
        else args.profile_timing,
        args.auto_embedding_batch_size_candidates,
        args.auto_embedding_batch_size_min,
        args.auto_embedding_batch_size_max,
        args.auto_embedding_batch_size_sample_size,
        args.auto_embedding_batch_size_memory_utilization,
        args.auto_reranker_batch_size_candidates,
        args.auto_reranker_batch_size_min,
        args.auto_reranker_batch_size_max,
        args.auto_reranker_batch_size_sample_size,
        args.auto_reranker_batch_size_memory_utilization,
        ranking_column=args.ranking_column,
        positive_score_column=args.positive_score_column,
    )
