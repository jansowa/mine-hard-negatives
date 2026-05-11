import argparse
import logging
import os
from datetime import datetime
from functools import partial

import pandas as pd
from decouple import config
from tqdm import tqdm

from batch_tuning import (
    OOMRetryReranker,
    benchmark_embedding_batch_size,
    parse_batch_size_candidates,
    set_dense_embedding_batch_size,
    validate_batch_size_options,
)
from batch_tuning import (
    benchmark_reranker_batch_size as benchmark_single_reranker_batch_size,
)
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


def _import_torch():
    import torch

    return torch


def get_negative_mining_stage_defaults() -> dict:
    return {
        "candidate_beta": config("CANDIDATE_BETA", cast=float, default=config("BETA", cast=float, default=0.01)),
        "candidate_u_floor": config(
            "CANDIDATE_U_FLOOR",
            cast=float,
            default=config("U_FLOOR", cast=float, default=0.005),
        ),
        "candidate_target": config("CANDIDATE_TARGET", cast=int, default=40),
        "candidate_search_chunk": config("CANDIDATE_SEARCH_CHUNK", cast=int, default=128),
        "candidate_max_offset_iters": config("CANDIDATE_MAX_OFFSET_ITERS", cast=int, default=10),
        "candidate_random_fallback": config("CANDIDATE_RANDOM_FALLBACK", cast=int, default=128),
    }


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


def benchmark_model_sets_reranker_batch_size(
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

        selected = benchmark_single_reranker_batch_size(
            model_set.reranker_tokenizer,
            model_set.reranker_model,
            sample_queries,
            sample_docs,
            candidates,
            rerank_function,
            memory_utilization=memory_utilization,
            device_id=gpu_id,
            label=f"reranker GPU {gpu_id}",
            logger=logger,
        )
        selected_by_gpu.append((gpu_id, selected))

    if not selected_by_gpu:
        return candidates[0]

    selected = min(batch_size for _, batch_size in selected_by_gpu)
    if len(selected_by_gpu) > 1:
        details = ", ".join(f"gpu{gpu_id}={batch_size}" for gpu_id, batch_size in selected_by_gpu)
        logger.info("Selected global reranker batch size: %s (%s)", selected, details)
    return selected


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
    top_k: int | None,
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
    candidate_beta: float = config("CANDIDATE_BETA", cast=float, default=config("BETA", cast=float, default=0.01)),
    candidate_u_floor: float = config(
        "CANDIDATE_U_FLOOR",
        cast=float,
        default=config("U_FLOOR", cast=float, default=0.005),
    ),
    candidate_target: int = config("CANDIDATE_TARGET", cast=int, default=40),
    candidate_search_chunk: int = config("CANDIDATE_SEARCH_CHUNK", cast=int, default=128),
    candidate_max_offset_iters: int = config("CANDIDATE_MAX_OFFSET_ITERS", cast=int, default=10),
    candidate_random_fallback: int = config("CANDIDATE_RANDOM_FALLBACK", cast=int, default=128),
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

    torch = _import_torch()
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    if not ranking_column:
        raise ValueError("--ranking_column must not be empty")
    if not positive_score_column:
        raise ValueError("--positive_score_column must not be empty")

    logger.info(f"{num_gpus} GPUs available for processing")
    if top_k is not None:
        logger.warning("--top_k/TOP_K is deprecated and ignored by iterative negative mining")
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
            logger=logger,
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
        beta=candidate_beta,
        u_floor=candidate_u_floor,
        candidate_target=candidate_target,
        candidate_search_chunk=candidate_search_chunk,
        candidate_max_offset_iters=candidate_max_offset_iters,
        candidate_random_fallback=candidate_random_fallback,
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
        reranker_batch_size = benchmark_model_sets_reranker_batch_size(
            model_sets,
            rerank_function,
            sample_queries,
            sample_docs,
            reranker_candidates,
            auto_reranker_batch_size_memory_utilization,
        )
    else:
        logger.info("Using explicit reranker batch size: %s", reranker_batch_size)

    assert reranker_batch_size is not None
    safe_rerank_function = OOMRetryReranker(rerank_function, reranker_batch_size, logger=logger)
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
    stage_defaults = get_negative_mining_stage_defaults()
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=config("TOP_K", cast=int, default=None),
        help="Deprecated compatibility option; iterative mining ignores this value.",
    )
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
    parser.add_argument("--candidate_beta", type=float, default=stage_defaults["candidate_beta"])
    parser.add_argument("--candidate_u_floor", type=float, default=stage_defaults["candidate_u_floor"])
    parser.add_argument("--candidate_target", type=int, default=stage_defaults["candidate_target"])
    parser.add_argument("--candidate_search_chunk", type=int, default=stage_defaults["candidate_search_chunk"])
    parser.add_argument("--candidate_max_offset_iters", type=int, default=stage_defaults["candidate_max_offset_iters"])
    parser.add_argument("--candidate_random_fallback", type=int, default=stage_defaults["candidate_random_fallback"])
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
        candidate_beta=args.candidate_beta,
        candidate_u_floor=args.candidate_u_floor,
        candidate_target=args.candidate_target,
        candidate_search_chunk=args.candidate_search_chunk,
        candidate_max_offset_iters=args.candidate_max_offset_iters,
        candidate_random_fallback=args.candidate_random_fallback,
    )
