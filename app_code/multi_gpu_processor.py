import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty  # Modified import
import time
from typing import List, Callable, Any, Dict, Tuple
import logging
import torch
import json
import os
import pandas as pd


class GPUModelSet:
    """Represents a set of models (dense, sparse, reranker) running on a specific GPU"""
    
    def __init__(self, gpu_id: int, reranker_tokenizer, reranker_model, logger=None):
        self.gpu_id = gpu_id
        self.reranker_tokenizer = reranker_tokenizer
        self.reranker_model = reranker_model
        self.processed_batches = 0
        self.total_queries = 0
        self.logger = logger or logging.getLogger(__name__)
    
    def process_query_batch(self, query_batch: List[Dict], vector_store, rerank_function: Callable, 
                           top_k: int, reranker_batch_size: int) -> List[Tuple]:
        """Process a batch of queries to find negatives (typically 1 query for line-by-line processing)"""
        
        results = []
        for query_data in query_batch:
            # Retrieve documents using the vector store
            retrieved_docs = [
                document for document in vector_store.as_retriever(search_kwargs={"k": top_k}).invoke(query_data['text'])
                if document.metadata['document_id'] != query_data['document_id']
            ]
            
            # Rerank the documents
            if retrieved_docs:
                ranking = rerank_function(
                    self.reranker_tokenizer, 
                    self.reranker_model, 
                    query_data['text'],
                    [document.page_content for document in retrieved_docs],
                    batch_size=reranker_batch_size
                )
                
                # Collect results
                for document, rank in zip(retrieved_docs, ranking):
                    results.append((query_data['query_id'], document.metadata['document_id'], rank))
        
        self.total_queries += len(query_batch)
        
        return results


class MultiGPUNegativeFinder:
    """Manages distribution of query processing across multiple GPU model sets"""
    
    def __init__(self, model_sets: List[GPUModelSet], 
                 output_path: str = None, progress_bar=None, logger=None, resume: bool = False):
        self.model_sets = model_sets
        self.query_queue = Queue()
        self.completed_batches = 0  # Now tracks completed queries
        self.total_batches = 0      # Now tracks total queries
        self.total_queries = 0
        self.processed_queries = 0
        self.queries_lock = threading.Lock()
        self.output_path = output_path
        self.progress_bar = progress_bar
        self.logger = logger or logging.getLogger(__name__)
        self.resume = resume
        self.start_time = None  # Track overall processing start time
        
        # Create individual worker output paths
        self.output_dir = os.path.dirname(output_path) if output_path else "."
        self.output_basename = os.path.splitext(os.path.basename(output_path))[0] if output_path else "negatives"
        self.worker_files = []
        for i, model_set in enumerate(self.model_sets):
            worker_file = os.path.join(self.output_dir, f"{self.output_basename}_worker_{model_set.gpu_id}_{i}.jsonl")
            self.worker_files.append(worker_file)
            # Clear any existing worker file only if not resuming
            if not resume and os.path.exists(worker_file):
                os.remove(worker_file)
    
    def get_processed_query_ids(self) -> set:
        """Get set of query IDs that have already been processed from existing worker files"""
        import json
        import os
        
        processed_query_ids = set()
        
        for worker_file in self.worker_files:
            if os.path.exists(worker_file):
                try:
                    with open(worker_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                result = json.loads(line.strip())
                                processed_query_ids.add(result['query_id'])
                except Exception as e:
                    self.logger.warning(f"Error reading worker file {worker_file}: {e}")
        
        if processed_query_ids:
            self.logger.info(f"[RESUME] Found {len(processed_query_ids)} already processed queries")
        
        return processed_query_ids
    
    def save_result_to_jsonl(self, worker_id: int, query_id: int, document_id: int, ranking: float) -> None:
        """Save a single result to worker's JSONL file"""
        
        result = {
            "query_id": int(query_id),
            "document_id": int(document_id), 
            "ranking": float(ranking)
        }
        
        try:
            with open(self.worker_files[worker_id], 'a') as f:
                f.write(json.dumps(result) + '\n')
        except Exception as e:
            self.logger.error(f"[WORKER {worker_id}] Error saving result to JSONL: {e}")
            raise
    
    def create_batches(self, queries: List[Dict]) -> None:
        """Add individual queries to queue for processing, filtering out already processed ones if resuming"""
        # Filter out already processed queries if resuming
        if self.resume:
            processed_query_ids = self.get_processed_query_ids()
            original_count = len(queries)
            queries = [query for query in queries if query['query_id'] not in processed_query_ids]
            filtered_count = original_count - len(queries)
            if filtered_count > 0:
                self.logger.info(f"[RESUME] Filtered out {filtered_count} already processed queries")
        
        self.total_batches = len(queries)  # Each query is now a "batch" of size 1
        self.total_queries = len(queries)
        
        for i, query in enumerate(queries):
            self.query_queue.put((i, [query]))  # Each "batch" contains one query
        
        self.logger.info(f"[MAIN] Created {self.total_batches} individual query tasks")
        self.logger.info(f"[MAIN] Total queries to process: {self.total_queries}")
    
    def worker(self, worker_id: int, model_set: GPUModelSet, vector_store, rerank_function: Callable, 
               top_k: int, reranker_batch_size: int) -> None:
        """Worker function for each GPU model set"""
        results_count = 0
        
        while True:
            try:
                # Get next query from queue (non-blocking with timeout)
                query_id, query_batch = self.query_queue.get(timeout=1)
                
                # Process the single query (query_batch contains only 1 query)
                start_time = time.time()
                batch_results = model_set.process_query_batch(
                    query_batch, vector_store, rerank_function, top_k, reranker_batch_size
                )
                processing_time = time.time() - start_time
                
                # Save each result immediately to worker's JSONL file
                for query_id_result, document_id, ranking in batch_results:
                    self.save_result_to_jsonl(worker_id, query_id_result, document_id, ranking)
                    results_count += 1
                
                # Update progress for the single query
                if self.progress_bar is not None:
                    self.progress_bar.update(1)
                
                # Update processed queries count and log progress every 10 queries
                with self.queries_lock:
                    self.processed_queries += 1
                    if self.processed_queries % 1000 == 0:
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time if self.start_time else 0
                        
                        # Calculate estimated time to finish
                        if self.processed_queries > 0:
                            queries_per_second = self.processed_queries / elapsed_time if elapsed_time > 0 else 0
                            remaining_queries = self.total_queries - self.processed_queries
                            eta_seconds = remaining_queries / queries_per_second if queries_per_second > 0 else 0
                            
                            # Format time strings
                            elapsed_str = f"{int(elapsed_time//3600)}h {int((elapsed_time%3600)//60)}m {int(elapsed_time%60)}s"
                            eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m {int(eta_seconds%60)}s"
                            
                            self.logger.info(f"[PROGRESS] Processed {self.processed_queries}/{self.total_queries} queries "
                                           f"({self.processed_queries/self.total_queries*100:.1f}%) | "
                                           f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                        else:
                            elapsed_str = f"{int(elapsed_time//3600)}h {int((elapsed_time%3600)//60)}m {int(elapsed_time%60)}s"
                            self.logger.info(f"[PROGRESS] Processed {self.processed_queries}/{self.total_queries} queries "
                                           f"({self.processed_queries/self.total_queries*100:.1f}%) | "
                                           f"Elapsed: {elapsed_str} | ETA: calculating...")
                
                # Update completion count
                self.completed_batches += 1
                
                self.logger.debug(f"[GPU {model_set.gpu_id}] Query {query_id} completed "
                          f"in {processing_time:.2f}s, saved {len(batch_results)} results "
                          f"({self.completed_batches}/{self.total_batches} total)")
                
                # Mark task as done
                self.query_queue.task_done()
                
            except Empty:
                self.logger.info(f"[GPU {model_set.gpu_id}] Worker found no more queries to process... quitting")
                break
            except Exception as e:
                self.logger.error(f"[GPU {model_set.gpu_id}] Worker error: {e}")
                break
        
        self.logger.info(f"[GPU {model_set.gpu_id}] Worker finished. Total results saved: {results_count}")
    
    def process_all(self, queries: List[Dict], vector_store, rerank_function: Callable, 
                   top_k: int, reranker_batch_size: int) -> int:
        """Process all queries using available GPU model sets"""
        self.logger.info(f"[MAIN] Starting processing of {len(queries)} queries across {len(self.model_sets)} GPUs")
        
        # Create batches
        self.create_batches(queries)
        
        # Start worker threads for each GPU model set
        threads = []
        for i, model_set in enumerate(self.model_sets):
            thread = threading.Thread(
                target=self.worker, 
                args=(i, model_set, vector_store, rerank_function, top_k, reranker_batch_size)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for all batches to be processed
        start_time = time.time()
        self.start_time = start_time  # Store for progress calculations
        
        try:
            self.query_queue.join()
        except KeyboardInterrupt:
            self.logger.info("[MAIN] Interrupted by user, saving remaining results...")
        except Exception as e:
            self.logger.error(f"[MAIN] Error during processing: {e}")
        
        total_time = time.time() - start_time
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)
        
        # Consolidate all worker files into final parquet
        if self.output_path:
            self.consolidate_worker_files()
        
        self.logger.info(f"[MAIN] All processing completed in {total_time:.2f}s")
        
        # Print summary statistics
        self.print_statistics(total_time)
        
        return self.completed_batches
    
    def consolidate_worker_files(self) -> None:
        """Consolidate all worker JSONL files into final parquet file"""
        
        self.logger.info("[MAIN] Consolidating worker files into final parquet...")
        
        all_results = []
        total_results = 0
        
        # Read all worker files (preserving them for manual recovery if needed)
        for i, worker_file in enumerate(self.worker_files):
            if os.path.exists(worker_file):
                worker_results = 0
                try:
                    with open(worker_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                result = json.loads(line.strip())
                                all_results.append(result)
                                worker_results += 1
                    
                    total_results += worker_results
                    self.logger.info(f"[MAIN] Worker {i}: {worker_results} results")
                    
                except Exception as e:
                    self.logger.error(f"[MAIN] Error reading worker file {worker_file}: {e}")
            else:
                self.logger.warning(f"[MAIN] Worker file {worker_file} not found")
        
        if all_results:
            try:
                # Convert to DataFrame and save as parquet
                df = pd.DataFrame(all_results)
                df = df.astype({"query_id": "int64", "document_id": "int64", "ranking": "float32"})
                df.to_parquet(self.output_path, index=False)
                self.logger.info(f"[MAIN] Saved {total_results} results to {self.output_path}")
                self.logger.info(f"[MAIN] Worker files preserved at: {', '.join(self.worker_files)}")
                        
            except Exception as e:
                self.logger.error(f"[MAIN] Error creating parquet file: {e}")
                self.logger.info(f"[MAIN] Worker files preserved for manual recovery")
                raise
        else:
            self.logger.warning("[MAIN] No results found to consolidate")
    
    def print_statistics(self, total_time: float) -> None:
        """Print processing statistics"""
        self.logger.info("[MAIN] === Processing Statistics ===")
        self.logger.info(f"[MAIN] Total processing time: {total_time:.2f}s")
        self.logger.info(f"[MAIN] Total queries processed: {self.completed_batches}")
        
        for model_set in self.model_sets:
            avg_time_per_query = total_time / model_set.total_queries if model_set.total_queries > 0 else 0
            self.logger.info(f"[GPU {model_set.gpu_id}] "
                      f"{model_set.total_queries} queries processed, "
                      f"avg {avg_time_per_query:.2f}s/query")


def should_resume_processing(output_path: str, model_sets: List[GPUModelSet]) -> bool:
    """Check if there are existing worker files that suggest we should resume processing"""
    
    if not output_path:
        return False
    
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    
    for i, model_set in enumerate(model_sets):
        worker_file = os.path.join(output_dir, f"{output_basename}_worker_{model_set.gpu_id}_{i}.jsonl")
        if os.path.exists(worker_file) and os.path.getsize(worker_file) > 0:
            return True
    
    return False


def setup_multi_gpu_models(dense_model_name: str, sparse_model_name: str, reranker_model_name: str,
                          embedding_batch_size: int, dense_prompt: str, models_per_gpu: int = 1, logger=None) -> List[GPUModelSet]:
    """Setup model sets across all available GPUs, with multiple model instances per GPU if desired"""
    from models import get_dense_model, get_sparse_model, get_reranker_model
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    
    logger.info(f"[MAIN] Setting up models across {num_gpus} GPUs, {models_per_gpu} model sets per GPU")
    
    model_sets = []
    for gpu_id in range(num_gpus):
        for instance in range(models_per_gpu):
            logger.info(f"[GPU {gpu_id}] Loading models (instance {instance+1}/{models_per_gpu})")
            # Load models on specific GPU
            # dense_model = get_dense_model(dense_model_name, batch_size=embedding_batch_size, 
            #                             prompt=dense_prompt, gpu_id=gpu_id)
            # sparse_model = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size, 
            #                               gpu_id=gpu_id)
            reranker_tokenizer, reranker_model = get_reranker_model(model_name=reranker_model_name, 
                                                                  gpu_id=gpu_id)
            model_set = GPUModelSet(gpu_id, reranker_tokenizer, reranker_model, logger)
            model_sets.append(model_set)
            logger.info(f"[GPU {gpu_id}] Successfully loaded models (instance {instance+1}/{models_per_gpu})")
    return model_sets