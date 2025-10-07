from models import get_dense_model, get_sparse_model, rerank
import argparse
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import pandas as pd
from decouple import config
from utils.vdb import get_qdrant_client
import torch
import logging
from datetime import datetime
from multi_gpu_processor import MultiGPUNegativeFinder, setup_multi_gpu_models, should_resume_processing
from tqdm import tqdm
import os

# Setup logging with same datetime-based filename as multi_gpu_processor
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{current_time}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Keep console output as well
    ]
)

# Suppress HTTP request logs from Qdrant client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from qdrant_client import models

class PatchedQdrantVectorStore(QdrantVectorStore):
    """This class is introduced to fix problem related to empty results with offset>0"""
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter=None, search_params=None,
        offset: int = 0, score_threshold=None, consistency=None,
        hybrid_fusion=None, **kwargs
    ):
        query_options = {
            "collection_name": self.collection_name,
            "query_filter": filter,
            "search_params": search_params,
            "limit": k,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
            "score_threshold": score_threshold,
            "consistency": consistency,
            **kwargs,
        }

        if self.retrieval_mode == RetrievalMode.HYBRID:
            q_dense = self.embeddings.embed_query(query)
            q_sparse = self.sparse_embeddings.embed_query(query)

            prefetch_limit = k + offset

            results = self.client.query_points(
                prefetch=[
                    models.Prefetch(
                        using=self.vector_name,
                        query=q_dense,
                        filter=filter,
                        limit=prefetch_limit,
                        params=search_params,
                    ),
                    models.Prefetch(
                        using=self.sparse_vector_name,
                        query=models.SparseVector(
                            indices=q_sparse.indices,
                            values=q_sparse.values,
                        ),
                        filter=filter,
                        limit=prefetch_limit,
                        params=search_params,
                    ),
                ],
                query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
                **query_options,
            ).points
        else:
            # oryginalne ścieżki DENSE/SPARSE bez zmian
            return super().similarity_search_with_score(
                query, k, filter, search_params, offset, score_threshold,
                consistency, hybrid_fusion, **kwargs
            )

        return [
            (
                self._document_from_point(
                    r, self.collection_name,
                    self.content_payload_key, self.metadata_payload_key,
                ),
                r.score,
            )
            for r in results
        ]


def find_negatives_multigpu(dense_model_name: str, sparse_model_name: str, embedding_batch_size: int,
                            reranker_model_name: str, reranker_batch_size: int,
                            collection_name: str, queries_path: str, relevant_path: str,
                            output_path: str, top_k: int, force_resume: bool = None):
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")

    logger.info(f"{num_gpus} GPUs available for processing")

    # --- Load dedicated models for vector store on GPU 0 ---
    dense_model_vs = get_dense_model(dense_model_name, batch_size=embedding_batch_size, prompt=config("DENSE_PROMPT"), gpu_id=0)
    sparse_model_vs = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size, gpu_id=0)
    
    # Setup multi-GPU models for processing (these can be on all GPUs)
    model_sets = setup_multi_gpu_models(
        reranker_model_name,
        models_per_gpu=1, logger=logger
    )
    
    # Check if we should resume processing
    if force_resume is not None:
        resume = force_resume
        logger.info(f"Resume mode forced to: {resume}")
    else:
        resume = should_resume_processing(output_path, model_sets)
    
    if resume:
        logger.info("Detected existing worker files - resuming processing")
    else:
        logger.info("Starting fresh processing")
    
    # Setup vector store using dedicated models (on GPU 0)
    logger.info("Setting up vector store")
    client = get_qdrant_client()
    vector_store = PatchedQdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_model_vs,  # Dedicated model for vector store
        sparse_embedding=sparse_model_vs,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )
    
    logger.info("Loading queries and relevance data")
    queries_df = pd.read_parquet(queries_path)
    relevant_df = pd.read_parquet(relevant_path)
    
    # Get best positive for each query
    best_relevant_df = relevant_df.loc[relevant_df.groupby('query_id')['positive_ranking'].idxmax()]
    print(f"{best_relevant_df.head()=}")
    positives_df = queries_df.merge(best_relevant_df, left_on='id', right_on="query_id").drop(columns="id")
    
    # Convert DataFrame to list of dictionaries for processing
    queries_list = []
    for _, row in positives_df.iterrows():
        queries_list.append({
            'query_id': row['query_id'],
            'document_id': row['document_id'],
            'positive_ranking': row['positive_ranking'],
            'text': row['text']
        })
    
    # queries_list = queries_list[:100]

    logger.info(f"Loaded {len(queries_list)} queries for processing")
    
    # Setup multi-GPU processor with individual worker files
    processor = MultiGPUNegativeFinder(
        model_sets, 
        output_path=output_path,
        progress_bar=None,  # We'll set this up after we know how many queries to process
        logger=logger,
        resume=resume
    )
    
    # Get the number of queries that will actually be processed (after filtering)
    if resume:
        processed_query_ids = processor.get_processed_query_ids()
        remaining_queries = len([q for q in queries_list if q['query_id'] not in processed_query_ids])
        logger.info(f"Resuming: {len(processed_query_ids)} already processed, {remaining_queries} remaining")
        progress_bar = tqdm(total=remaining_queries, desc="Queries Processed", position=0)
    else:
        progress_bar = tqdm(total=len(queries_list), desc="Queries Processed", position=0)
    
    # Set the progress bar in the processor
    processor.progress_bar = progress_bar
    
    # Process all queries with individual worker files
    logger.info("Starting multi-GPU negative finding with individual worker files")
    try:
        completed_batches = processor.process_all(
            queries_list, vector_store, rerank, top_k, reranker_batch_size
        )
        logger.info(f"Successfully processed {completed_batches} batches")
        progress_bar.close()
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user, but progress has been saved")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.info("Some progress may have been saved to the output file")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find negative samples using multi-GPU processing")
    parser.add_argument("--dense_model_name", type=str, required=False, 
                       help="Name of dense model to calculate embeddings", 
                       default=config("DENSE_EMBEDDER_NAME"))
    parser.add_argument("--sparse_model_name", type=str, required=False, 
                       help="Name of sparse model to calculate embeddings", 
                       default=config("SPLADE_MODEL_NAME"))
    parser.add_argument("--embedding_batch_size", type=int, required=False, 
                       help="Number of documents in one embeddings model batch", 
                       default=config("EMBEDDER_BATCH_SIZE", cast=int))
    parser.add_argument("--reranker_model_name", type=str, required=False, 
                       help="Name of reranker model", 
                       default=config("RERANKER_NAME"))
    parser.add_argument("--reranker_batch_size", type=int, required=False, 
                       help="Number of documents in one reranker batch", 
                       default=config("RERANKER_BATCH_SIZE", cast=int))
    parser.add_argument("--database_collection_name", type=str, required=False, 
                       help="Name of database collection", default="all_documents")
    parser.add_argument("--queries_path", type=str, required=False,
                       help="Path to the queries parquet file.", 
                       default=config("QUERIES_PATH"))
    parser.add_argument("--relevant_path", type=str, required=False, 
                       help="Path to the relevancy parquet file.", 
                       default=config("RELEVANT_WITH_SCORE_PATH"))
    parser.add_argument("--output_path", type=str, required=False, 
                       help="Path to the output parquet file.", 
                       default=config("NEGATIVES_PATH"))
    parser.add_argument("--top_k", type=int, default=config("TOP_K", cast=int), 
                       required=False, help="Number of documents to retrieve")
    parser.add_argument("--resume", action="store_true", default=None,
                       help="Force resume from existing worker files")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None,
                       help="Force fresh start, ignoring existing worker files")
    
    args = parser.parse_args()
    
    find_negatives_multigpu(
        args.dense_model_name, args.sparse_model_name, args.embedding_batch_size, 
        args.reranker_model_name, args.reranker_batch_size,
        args.database_collection_name, args.queries_path,
        args.relevant_path, args.output_path, args.top_k, args.resume
    )