import json
import logging
import os
import threading
import time
from collections import defaultdict
from queue import Empty, Queue
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from decouple import config


class TimingStats:
    """Thread-safe aggregate timings, enabled only when explicitly requested."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"seconds": 0.0, "calls": 0, "items": 0})

    def record(self, name: str, seconds: float, items: int = 0) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._stats[name]
            row["seconds"] += float(seconds)
            row["calls"] += 1
            row["items"] += int(items)

    def add_items(self, name: str, items: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            row = self._stats[name]
            row["items"] += int(items)

    def log(self, logger: logging.Logger) -> None:
        if not self.enabled:
            return
        with self._lock:
            snapshot = {
                name: dict(values)
                for name, values in sorted(self._stats.items())
            }

        if not snapshot:
            logger.info("[TIMING] No timing samples collected")
            return

        logger.info("[TIMING] === Hot Path Timings ===")
        for name, values in snapshot.items():
            seconds = values["seconds"]
            calls = values["calls"]
            items = values["items"]
            bits = [f"[TIMING] {name}: {seconds:.3f}s", f"calls={calls}"]
            if calls:
                bits.append(f"avg_call={seconds / calls:.6f}s")
            if items:
                bits.append(f"items={items}")
                if seconds > 0:
                    bits.append(f"items/s={items / seconds:.1f}")
            logger.info(" | ".join(bits))


def _now_if_enabled(timing_stats: TimingStats | None) -> float | None:
    if timing_stats is not None and timing_stats.enabled:
        return time.perf_counter()
    return None


def _record_elapsed(timing_stats: TimingStats | None, name: str, started_at: float | None, items: int = 0) -> None:
    if started_at is not None and timing_stats is not None:
        timing_stats.record(name, time.perf_counter() - started_at, items=items)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _result_value(row: dict, key: str, default=None):
    value = row.get(key, default)
    if value is None:
        return default
    return value



class GPUModelSet:
    """Represents a set of models (dense, sparse, reranker) running on a specific GPU"""
    
    def __init__(self, gpu_id: int, reranker_tokenizer, reranker_model, relevant_path: str, logger=None,
                 skip_if_not_in_top_k: int = config("SKIP_IF_NOT_IN_TOP_K", cast=int, default=100),
                 beta: float = 0.4, u_floor: float = 0.1,
                 positive_score_column: str = "positive_ranking"):

        self.gpu_id = gpu_id
        self.reranker_tokenizer = reranker_tokenizer
        self.reranker_model = reranker_model
        self.processed_batches = 0
        self.total_queries = 0
        self.logger = logger or logging.getLogger(__name__)
        self.skip_if_not_in_top_k = skip_if_not_in_top_k
        self.positive_score_column = positive_score_column

        relevant_df = pd.read_parquet(relevant_path)

        required_columns = {"query_id", "document_id", positive_score_column}
        if not required_columns.issubset(relevant_df.columns):
            missing = ", ".join(sorted(required_columns - set(relevant_df.columns)))
            raise ValueError(f"Required columns are missing in Parquet for 'relevant with score': {missing}")

        scores = np.sort(relevant_df[positive_score_column].dropna().to_numpy(copy=True))
        if len(scores) == 0:
            raise ValueError(f"No non-null values found in positive score column {positive_score_column!r}")
        self._ecdf_x = scores
        n = len(scores)
        self._ecdf_y = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])

        self.pos_score_by_qd = {
            (str(row["query_id"]), str(row["document_id"])): float(row[positive_score_column])
            for _, row in relevant_df.iterrows()
        }

        self.beta = beta
        self.u_floor = u_floor

    def _percentile(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self._ecdf_x, self._ecdf_y, left=0.0, right=1.0)

    def _is_negative_percentile(self, u_pos: float, u_doc: float) -> bool:
        return (u_doc - u_pos) <= -self.beta or (u_doc <= self.u_floor)

    def _search_with_offset(self, backend, query_text: str, limit: int, offset: int):
        return backend.search(query_text=query_text, k=limit, offset=offset)

    def _search_many_offsets(self, backend, query_text: str, limit: int, offsets: List[int]):
        if hasattr(backend, "search_many_offsets"):
            return backend.search_many_offsets(query_text=query_text, k=limit, offsets=offsets)
        return {
            offset: self._search_with_offset(backend, query_text, limit, offset)
            for offset in offsets
        }

    def _random_sample(self, backend, limit: int):
        return backend.random_sample(k=limit)

    def _positive_score(self, query_data: Dict) -> float:
        if "positive_score" in query_data:
            return float(query_data["positive_score"])
        positive_score_column = getattr(self, "positive_score_column", "positive_ranking")
        if positive_score_column in query_data:
            return float(query_data[positive_score_column])
        return float(query_data["positive_ranking"])

    def _build_result(self, state: Dict, document, score: float, percentile: float, selected: bool) -> dict:
        metadata = getattr(document, "metadata", {}) or {}
        return {
            "query_id": state["qid"],
            "document_id": str(metadata.get("document_id", "")),
            "ranking": float(score),
            "candidate_percentile": float(percentile),
            "candidate_selected": bool(selected),
            "retrieval_rank": _optional_int(metadata.get("retrieval_rank")),
            "retrieval_offset": _optional_int(metadata.get("retrieval_offset")),
            "retrieval_score": _optional_float(metadata.get("retrieval_score")),
            "retrieval_source": metadata.get("retrieval_source"),
        }

    def process_query_batch(self, query_batch: List[Dict], backend, rerank_function: Callable,
                            reranker_batch_size: int,
                            timing_stats: TimingStats | None = None) -> List[Dict]:
        """
        Iteratively fetches documents in batches of 128, with an offset of 2^(n+7), n = 0..9,
        until it collects ≥20 negatives according to the percentile rule (beta = 0.4, u_floor = 0.1),
        or completes 10 iterations; then it fetches an additional random 128 and stops.
        Returns every scored candidate collected during the process for each query.
        """
        NEG_TARGET = 20
        CHUNK = 128
        MAX_ITERS = 10
        OFFSET_GROUP_SIZE = config("SEARCH_MANY_OFFSETS_GROUP_SIZE", cast=int, default=2)

        offsets = [0 if n == 0 else 2 ** (n + 7) for n in range(MAX_ITERS)]
        offset_groups = [
            offsets[i:i + max(1, OFFSET_GROUP_SIZE)]
            for i in range(0, len(offsets), max(1, OFFSET_GROUP_SIZE))
        ]
        states = []

        for query_data in query_batch:
            qtext = query_data["text"]
            qid = str(query_data["query_id"])
            pos_doc_id = str(query_data["document_id"])

            s_pos = self._positive_score(query_data)
            u_pos = float(self._percentile(np.array([s_pos]))[0])

            states.append({
                "qtext": qtext,
                "qid": qid,
                "u_pos": u_pos,
                "seen_ids": set([pos_doc_id]),
                "collected_docs": [],
                "neg_count": 0,
                "done": False,
                "docs_by_offset": {},
            })

        for offset_group in offset_groups:
            for state in states:
                if not state["done"]:
                    started_at = _now_if_enabled(timing_stats)
                    state["docs_by_offset"].update(
                        self._search_many_offsets(backend, state["qtext"], CHUNK, offset_group)
                    )
                    _record_elapsed(timing_stats, "search_many_offsets", started_at, items=len(offset_group))

            for offset in offset_group:
                rerank_queries = []
                rerank_docs = []
                rerank_state_doc = []

                for state in states:
                    if state["done"]:
                        continue

                    batch_docs = []
                    for d in state["docs_by_offset"].get(offset, []):
                        did = d.metadata.get("document_id")
                        if did is None:
                            continue
                        did = str(did)
                        if did in state["seen_ids"]:
                            continue
                        state["seen_ids"].add(did)
                        batch_docs.append(d)

                    for d in batch_docs:
                        rerank_queries.append(state["qtext"])
                        rerank_docs.append(d.page_content)
                        rerank_state_doc.append((state, d))

                if not rerank_docs:
                    continue

                started_at = _now_if_enabled(timing_stats)
                batch_scores = rerank_function(
                    self.reranker_tokenizer,
                    self.reranker_model,
                    rerank_queries,
                    rerank_docs,
                    batch_size=reranker_batch_size
                )
                _record_elapsed(timing_stats, "rerank", started_at, items=len(rerank_docs))

                u_docs = self._percentile(np.array(batch_scores, dtype=float))
                for (state, d), s, u_d in zip(rerank_state_doc, batch_scores, u_docs):
                    selected = self._is_negative_percentile(state["u_pos"], float(u_d))
                    state["collected_docs"].append(self._build_result(state, d, float(s), float(u_d), selected))
                    if selected:
                        state["neg_count"] += 1

                for state in states:
                    if state["neg_count"] >= NEG_TARGET:
                        state["done"] = True

            if all(state["done"] for state in states):
                break

        random_queries = []
        random_docs = []
        random_state_doc = []
        for state in states:
            if state["neg_count"] >= NEG_TARGET:
                continue
            started_at = _now_if_enabled(timing_stats)
            rand_docs = self._random_sample(backend, CHUNK)
            _record_elapsed(timing_stats, "random_sample", started_at, items=len(rand_docs))
            for d in rand_docs:
                did = str(d.metadata.get("document_id", ""))
                if did in state["seen_ids"]:
                    continue
                state["seen_ids"].add(did)
                random_queries.append(state["qtext"])
                random_docs.append(d.page_content)
                random_state_doc.append((state, d))

        if random_docs:
            started_at = _now_if_enabled(timing_stats)
            rand_scores = rerank_function(
                self.reranker_tokenizer,
                self.reranker_model,
                random_queries,
                random_docs,
                batch_size=reranker_batch_size
            )
            _record_elapsed(timing_stats, "rerank_random", started_at, items=len(random_docs))
            u_docs = self._percentile(np.array(rand_scores, dtype=float))
            for (state, d), s, u_d in zip(random_state_doc, rand_scores, u_docs):
                selected = self._is_negative_percentile(state["u_pos"], float(u_d))
                state["collected_docs"].append(self._build_result(state, d, float(s), float(u_d), selected))

        results: List[Dict] = []
        for state in states:
            results.extend(state["collected_docs"])

        self.total_queries += len(query_batch)
        return results


class MultiGPUNegativeFinder:
    """Manages distribution of query processing across multiple GPU model sets"""
    
    def __init__(self, model_sets: List[GPUModelSet], 
                 output_path: str = None, progress_bar=None, logger=None, resume: bool = False,
                 profile_timing: bool = False, ranking_column: str = "ranking"):
        if not ranking_column:
            raise ValueError("ranking_column must not be empty")
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
        self.timing_stats = TimingStats(enabled=profile_timing)
        self.parquet_row_group_size = max(1, config("NEGATIVES_PARQUET_ROW_GROUP_SIZE", cast=int, default=100_000))
        self.ranking_column = ranking_column
        self.extended_output = ranking_column != "ranking"
        
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
        with open(self.worker_files[worker_id], 'a') as f:
            self.write_results_to_jsonl(f, [(query_id, document_id, ranking)])

    def _normalise_result(self, result: dict | tuple) -> dict:
        if isinstance(result, dict):
            score = result.get(self.ranking_column, result.get("ranking"))
            row = {
                "query_id": str(result["query_id"]),
                "document_id": str(result["document_id"]),
                self.ranking_column: float(score),
            }
            if self.extended_output:
                row.update({
                    "candidate_percentile": _optional_float(result.get("candidate_percentile")),
                    "candidate_selected": bool(result.get("candidate_selected", False)),
                    "retrieval_rank": _optional_int(result.get("retrieval_rank")),
                    "retrieval_offset": _optional_int(result.get("retrieval_offset")),
                    "retrieval_score": _optional_float(result.get("retrieval_score")),
                    "retrieval_source": _result_value(result, "retrieval_source"),
                })
            return row

        query_id, document_id, ranking = result[:3]
        return {
            "query_id": str(query_id),
            "document_id": str(document_id),
            self.ranking_column: float(ranking),
        }

    def write_results_to_jsonl(self, handle, results: List[dict | tuple]) -> None:
        """Write a batch of results to an already-open JSONL handle."""
        if not results:
            return

        started_at = _now_if_enabled(self.timing_stats)
        lines = []
        for result in results:
            lines.append(json.dumps(self._normalise_result(result)) + '\n')

        try:
            handle.writelines(lines)
            _record_elapsed(self.timing_stats, "jsonl_write", started_at, items=len(results))
        except Exception as e:
            self.logger.error(f"Error saving results to JSONL: {e}")
            raise
    
    def create_batches(self, queries: List[Dict], query_batch_size: int = 1) -> None:
        """Add individual queries to queue for processing, filtering out already processed ones if resuming"""
        # Filter out already processed queries if resuming
        if self.resume:
            processed_query_ids = self.get_processed_query_ids()
            original_count = len(queries)
            queries = [query for query in queries if query['query_id'] not in processed_query_ids]
            filtered_count = original_count - len(queries)
            if filtered_count > 0:
                self.logger.info(f"[RESUME] Filtered out {filtered_count} already processed queries")
        
        query_batch_size = max(1, query_batch_size)
        self.total_batches = (len(queries) + query_batch_size - 1) // query_batch_size
        self.total_queries = len(queries)
        
        for i in range(0, len(queries), query_batch_size):
            self.query_queue.put((i, queries[i:i + query_batch_size]))
        
        self.logger.info(f"[MAIN] Created {self.total_batches} query tasks (batch size {query_batch_size})")
        self.logger.info(f"[MAIN] Total queries to process: {self.total_queries}")
    
    def worker(self, worker_id: int, model_set: GPUModelSet, vector_store, rerank_function: Callable, 
               top_k: int, reranker_batch_size: int) -> None:
        """Worker function for each GPU model set"""
        results_count = 0

        with open(self.worker_files[worker_id], 'a') as output_handle:
            while True:
                try:
                    # Get next query from queue (non-blocking with timeout)
                    query_id, query_batch = self.query_queue.get(timeout=1)

                    start_time = time.time()
                    batch_results = model_set.process_query_batch(
                        query_batch,
                        vector_store,
                        rerank_function,
                        reranker_batch_size,
                        timing_stats=self.timing_stats,
                    )
                    processing_time = time.time() - start_time

                    self.write_results_to_jsonl(output_handle, batch_results)
                    results_count += len(batch_results)

                    # Update progress for the query batch
                    if self.progress_bar is not None:
                        self.progress_bar.update(len(query_batch))

                    # Update processed queries count and log progress every 1000 queries
                    with self.queries_lock:
                        self.processed_queries += len(query_batch)
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

                    self.completed_batches += 1

                    self.logger.debug(f"[GPU {model_set.gpu_id}] Query {query_id} completed "
                              f"in {processing_time:.2f}s, saved {len(batch_results)} results "
                              f"({self.completed_batches}/{self.total_batches} total)")

                    self.query_queue.task_done()

                except Empty:
                    self.logger.info(f"[GPU {model_set.gpu_id}] Worker found no more queries to process... quitting")
                    break
                except Exception:
                    self.logger.error(f"[GPU {model_set.gpu_id}] Worker error:", exc_info=True)
                    break
        
        self.logger.info(f"[GPU {model_set.gpu_id}] Worker finished. Total results saved: {results_count}")
    
    def process_all(self, queries: List[Dict], vector_store, rerank_function: Callable,
                   top_k: int, reranker_batch_size: int, query_batch_size: int = 1) -> int:
        """Process all queries using available GPU model sets"""
        self.logger.info(f"[MAIN] Starting processing of {len(queries)} queries across {len(self.model_sets)} GPUs")
        
        self.create_batches(queries, query_batch_size=query_batch_size)
        
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
        self.start_time = start_time

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

        total_results = 0
        temp_output_path = f"{self.output_path}.tmp"
        writer = None

        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([
            ("query_id", pa.string()),
            ("document_id", pa.string()),
            (self.ranking_column, pa.float32()),
        ])
        if self.extended_output:
            schema = pa.schema([
                ("query_id", pa.string()),
                ("document_id", pa.string()),
                (self.ranking_column, pa.float32()),
                ("candidate_percentile", pa.float32()),
                ("candidate_selected", pa.bool_()),
                ("retrieval_rank", pa.int32()),
                ("retrieval_offset", pa.int32()),
                ("retrieval_score", pa.float32()),
                ("retrieval_source", pa.string()),
            ])

        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        def write_buffer(rows: list[dict]) -> None:
            nonlocal writer, total_results
            if not rows:
                return
            started_at = _now_if_enabled(self.timing_stats)
            table = pa.table(
                {
                    "query_id": pa.array([row["query_id"] for row in rows], type=pa.string()),
                    "document_id": pa.array([row["document_id"] for row in rows], type=pa.string()),
                    self.ranking_column: pa.array([row[self.ranking_column] for row in rows], type=pa.float32()),
                    **({
                        "candidate_percentile": pa.array(
                            [row.get("candidate_percentile") for row in rows],
                            type=pa.float32(),
                        ),
                        "candidate_selected": pa.array(
                            [row.get("candidate_selected", False) for row in rows],
                            type=pa.bool_(),
                        ),
                        "retrieval_rank": pa.array(
                            [row.get("retrieval_rank") for row in rows],
                            type=pa.int32(),
                        ),
                        "retrieval_offset": pa.array(
                            [row.get("retrieval_offset") for row in rows],
                            type=pa.int32(),
                        ),
                        "retrieval_score": pa.array(
                            [row.get("retrieval_score") for row in rows],
                            type=pa.float32(),
                        ),
                        "retrieval_source": pa.array(
                            [row.get("retrieval_source") for row in rows],
                            type=pa.string(),
                        ),
                    } if self.extended_output else {}),
                },
                schema=schema,
            )
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, schema=schema, compression="zstd", use_dictionary=True)
            writer.write_table(table)
            total_results += len(rows)
            _record_elapsed(self.timing_stats, "parquet_write", started_at, items=len(rows))
        
        try:
            # Read all worker files (preserving them for manual recovery if needed)
            for i, worker_file in enumerate(self.worker_files):
                if os.path.exists(worker_file):
                    worker_results = 0
                    buffer = []
                    try:
                        with open(worker_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    result = json.loads(line.strip())
                                    buffer.append(self._normalise_result(result))
                                    worker_results += 1
                                    if len(buffer) >= self.parquet_row_group_size:
                                        write_buffer(buffer)
                                        buffer = []
                        write_buffer(buffer)

                        self.logger.info(f"[MAIN] Worker {i}: {worker_results} results")

                    except Exception as e:
                        self.logger.error(f"[MAIN] Error reading worker file {worker_file}: {e}")
                        raise
                else:
                    self.logger.warning(f"[MAIN] Worker file {worker_file} not found")

            if writer is not None:
                writer.close()
                os.replace(temp_output_path, self.output_path)
                self.logger.info(f"[MAIN] Saved {total_results} results to {self.output_path}")
                self.logger.info(f"[MAIN] Worker files preserved at: {', '.join(self.worker_files)}")
            else:
                self.logger.warning("[MAIN] No results found to consolidate")
        except Exception as e:
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            self.logger.error(f"[MAIN] Error creating parquet file: {e}")
            self.logger.info("[MAIN] Worker files preserved for manual recovery")
            raise
    
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

        self.timing_stats.log(self.logger)


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


def setup_multi_gpu_models(reranker_model_name: str,
                           relevant_path: str,
                           models_per_gpu: int = 1, logger=None,
                           positive_score_column: str = "positive_ranking") -> List[GPUModelSet]:
    """Setup model sets across all available GPUs, with multiple model instances per GPU if desired"""
    from models import get_reranker_model
    
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
            reranker_tokenizer, reranker_model = get_reranker_model(model_name=reranker_model_name,
                                                                  gpu_id=gpu_id)
            model_set = GPUModelSet(
                gpu_id,
                reranker_tokenizer,
                reranker_model,
                relevant_path,
                logger,
                positive_score_column=positive_score_column,
            )
            model_sets.append(model_set)
            logger.info(f"[GPU {gpu_id}] Successfully loaded models (instance {instance+1}/{models_per_gpu})")
    return model_sets
