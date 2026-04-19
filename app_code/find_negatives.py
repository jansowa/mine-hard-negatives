import argparse
import logging
import os
from datetime import datetime

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


def find_negatives_multigpu(
    dense_model_name: str,
    sparse_model_name: str,
    embedding_batch_size: int,
    reranker_model_name: str,
    reranker_batch_size: int,
    collection_name: str,
    queries_path: str,
    relevant_path: str,
    output_path: str,
    top_k: int,
    force_resume: bool = None,
    query_batch_size: int = config("NEGATIVE_QUERY_BATCH_SIZE", cast=int, default=4),
    profile_timing: bool = config("NEGATIVE_PROFILE_TIMING", cast=bool, default=False),
):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")

    logger.info(f"{num_gpus} GPUs available for processing")

    dense_model_vs = get_dense_model(
        dense_model_name,
        batch_size=embedding_batch_size,
        prompt=config("DENSE_PROMPT"),
        gpu_id=0,
    )

    backend_type = get_vector_db_backend()
    sparse_model_vs = None
    dense_dim_size = None
    if backend_type == "qdrant":
        sparse_model_vs = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size, gpu_id=0)
        dense_dim_size = len(dense_model_vs.embed_query("text"))

    model_sets = setup_multi_gpu_models(
        reranker_model_name,
        relevant_path,
        models_per_gpu=1,
        logger=logger,
    )

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

    logger.info("Loading queries and relevance data")
    queries_df = pd.read_parquet(queries_path)
    relevant_df = pd.read_parquet(relevant_path)

    queries_df["id"] = queries_df["id"].astype("string")
    relevant_df["query_id"] = relevant_df["query_id"].astype("string")
    relevant_df["document_id"] = relevant_df["document_id"].astype("string")

    best_relevant_df = relevant_df.loc[relevant_df.groupby("query_id")["positive_ranking"].idxmax()]
    positives_df = queries_df.merge(best_relevant_df, left_on="id", right_on="query_id").drop(columns="id")

    queries_list = [
        {
            "query_id": row["query_id"],
            "document_id": row["document_id"],
            "positive_ranking": row["positive_ranking"],
            "text": row["text"],
        }
        for _, row in positives_df.iterrows()
    ]

    logger.info(f"Loaded {len(queries_list)} queries for processing")

    processor = MultiGPUNegativeFinder(
        model_sets,
        output_path=output_path,
        progress_bar=None,
        logger=logger,
        resume=resume,
        profile_timing=profile_timing,
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
            rerank,
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
    parser.add_argument("--embedding_batch_size", type=int, default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--reranker_model_name", type=str, default=config("RERANKER_NAME"))
    parser.add_argument("--reranker_batch_size", type=int, default=config("RERANKER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, default="all_documents")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"))
    parser.add_argument("--output_path", type=str, default=config("NEGATIVES_PATH"))
    parser.add_argument("--top_k", type=int, default=config("TOP_K", cast=int))
    parser.add_argument("--query_batch_size", type=int, default=config("NEGATIVE_QUERY_BATCH_SIZE", cast=int, default=4))
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
        config("NEGATIVE_PROFILE_TIMING", cast=bool, default=False) if args.profile_timing is None else args.profile_timing,
    )
