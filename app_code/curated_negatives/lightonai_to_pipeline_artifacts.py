from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from decouple import UndefinedValueError, config

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from curated_negatives.flag_embedding import ensure_parent_dir
    from curated_negatives.lightonai_to_flag_embedding import DEFAULT_DATASET_NAME, _load_lightonai_component
else:
    from .flag_embedding import ensure_parent_dir
    from .lightonai_to_flag_embedding import DEFAULT_DATASET_NAME, _load_lightonai_component


def parse_splits(raw_splits: str) -> list[str]:
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("At least one LightOn split is required")
    return splits


def first_split(raw_splits: str) -> str:
    return parse_splits(raw_splits)[0]


def split_output_dir(output_root: str, split: str) -> str:
    return os.path.join(output_root, split)


def pipeline_artifact_paths(output_dir: str) -> list[str]:
    return [
        os.path.join(output_dir, "queries.parquet"),
        os.path.join(output_dir, "corpus.parquet"),
        os.path.join(output_dir, "relevant.parquet"),
        os.path.join(output_dir, "negative_candidates.parquet"),
    ]


def pipeline_artifacts_exist(output_dir: str) -> bool:
    return all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in pipeline_artifact_paths(output_dir))


def _config_optional(name: str, *, cast=None, default=None):
    try:
        value = config(name, default=None)
    except UndefinedValueError:
        return default
    if value in {None, ""}:
        return default
    if cast is None:
        return value
    return cast(value)


def _id_text_frame(dataset: datasets.Dataset, id_column: str, text_column: str) -> pd.DataFrame:
    frame = dataset.select_columns([id_column, text_column]).to_pandas()
    frame = frame.rename(columns={id_column: "id", text_column: "text"})
    frame["id"] = frame["id"].astype("string")
    frame["text"] = frame["text"].fillna("").astype("string")
    return frame


def _write_id_text_dataset(
    dataset: datasets.Dataset,
    output_path: str,
    id_column: str,
    text_column: str,
    batch_size: int = 100_000,
) -> int:
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    temp_path = f"{output_path}.tmp-{uuid.uuid4().hex}"
    writer = pq.ParquetWriter(temp_path, schema, compression="zstd", use_dictionary=True)
    row_count = 0
    try:
        for batch in dataset.iter(batch_size=batch_size):
            ids = [str(value) for value in batch[id_column]]
            texts = ["" if value is None else str(value) for value in batch[text_column]]
            writer.write_table(pa.Table.from_pydict({"id": ids, "text": texts}, schema=schema))
            row_count += len(ids)
        writer.close()
        os.replace(temp_path, output_path)
    except Exception:
        writer.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return row_count


def build_lightonai_pipeline_artifact_frames(
    queries: datasets.Dataset,
    documents: datasets.Dataset,
    scores: datasets.Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_frame = _id_text_frame(queries, "query_id", "query")
    corpus_frame = _id_text_frame(documents, "document_id", "document")

    relevant_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in scores:
        query_id = str(row["query_id"])
        document_ids = row["document_ids"]
        lightonai_scores = row["scores"]
        if not document_ids or not lightonai_scores:
            continue

        positive_id = str(document_ids[0])
        relevant_rows.append(
            {
                "query_id": query_id,
                "document_id": positive_id,
                "lightonai_positive_score": float(lightonai_scores[0]),
            }
        )

        seen_negative_ids: set[str] = set()
        for retrieval_rank, (document_id, score) in enumerate(zip(document_ids[1:], lightonai_scores[1:])):
            negative_id = str(document_id)
            if negative_id == positive_id or negative_id in seen_negative_ids:
                continue
            seen_negative_ids.add(negative_id)
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "document_id": negative_id,
                    "candidate_ranking": float(score),
                    "lightonai_score": float(score),
                    "candidate_selected": True,
                    "retrieval_rank": int(retrieval_rank),
                    "retrieval_source": "lightonai",
                }
            )

    relevant_frame = pd.DataFrame(
        relevant_rows,
        columns=["query_id", "document_id", "lightonai_positive_score"],
    )
    candidates_frame = pd.DataFrame(
        candidate_rows,
        columns=[
            "query_id",
            "document_id",
            "candidate_ranking",
            "lightonai_score",
            "candidate_selected",
            "retrieval_rank",
            "retrieval_source",
        ],
    )
    if not relevant_frame.empty:
        relevant_frame["query_id"] = relevant_frame["query_id"].astype("string")
        relevant_frame["document_id"] = relevant_frame["document_id"].astype("string")
    if not candidates_frame.empty:
        candidates_frame["query_id"] = candidates_frame["query_id"].astype("string")
        candidates_frame["document_id"] = candidates_frame["document_id"].astype("string")
        candidates_frame["candidate_selected"] = candidates_frame["candidate_selected"].astype(bool)
    return query_frame, corpus_frame, relevant_frame, candidates_frame


def write_lightonai_pipeline_artifacts(
    output_dir: str,
    queries: datasets.Dataset,
    documents: datasets.Dataset,
    scores: datasets.Dataset,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    query_count = _write_id_text_dataset(
        queries,
        os.path.join(output_dir, "queries.parquet"),
        "query_id",
        "query",
    )
    corpus_count = _write_id_text_dataset(
        documents,
        os.path.join(output_dir, "corpus.parquet"),
        "document_id",
        "document",
    )

    relevant_schema = pa.schema(
        [
            ("query_id", pa.string()),
            ("document_id", pa.string()),
            ("lightonai_positive_score", pa.float64()),
        ]
    )
    candidate_schema = pa.schema(
        [
            ("query_id", pa.string()),
            ("document_id", pa.string()),
            ("candidate_ranking", pa.float64()),
            ("lightonai_score", pa.float64()),
            ("candidate_selected", pa.bool_()),
            ("retrieval_rank", pa.int64()),
            ("retrieval_source", pa.string()),
        ]
    )
    relevant_path = os.path.join(output_dir, "relevant.parquet")
    candidates_path = os.path.join(output_dir, "negative_candidates.parquet")
    relevant_temp = f"{relevant_path}.tmp-{uuid.uuid4().hex}"
    candidates_temp = f"{candidates_path}.tmp-{uuid.uuid4().hex}"
    relevant_writer = pq.ParquetWriter(relevant_temp, relevant_schema, compression="zstd", use_dictionary=True)
    candidates_writer = pq.ParquetWriter(candidates_temp, candidate_schema, compression="zstd", use_dictionary=True)
    relevant_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    relevant_count = 0
    candidate_count = 0
    flush_rows = 100_000
    current_query_id: str | None = None
    current_positive_ids: set[str] = set()
    current_candidates: dict[str, dict[str, Any]] = {}
    completed_query_ids: set[str] = set()

    def flush_relevant() -> None:
        nonlocal relevant_rows
        if relevant_rows:
            relevant_writer.write_table(pa.Table.from_pylist(relevant_rows, schema=relevant_schema))
            relevant_rows = []

    def flush_candidates() -> None:
        nonlocal candidate_rows
        if candidate_rows:
            candidates_writer.write_table(pa.Table.from_pylist(candidate_rows, schema=candidate_schema))
            candidate_rows = []

    def flush_candidate_query() -> None:
        nonlocal candidate_rows, candidate_count, current_positive_ids, current_candidates
        if current_query_id is None:
            return
        completed_query_ids.add(current_query_id)
        rows = [
            row
            for document_id, row in current_candidates.items()
            if document_id not in current_positive_ids
        ]
        rows.sort(
            key=lambda row: (
                -float(row["candidate_ranking"]),
                int(row["retrieval_rank"]),
                str(row["document_id"]),
            )
        )
        candidate_count += len(rows)
        candidate_rows.extend(rows)
        if len(candidate_rows) >= flush_rows:
            flush_candidates()
        current_positive_ids = set()
        current_candidates = {}

    try:
        for batch in scores.iter(batch_size=256):
            for query_id, document_ids, lightonai_scores in zip(
                batch["query_id"],
                batch["document_ids"],
                batch["scores"],
            ):
                if not document_ids or not lightonai_scores:
                    continue
                query_id = str(query_id)
                if query_id != current_query_id:
                    flush_candidate_query()
                    if query_id in completed_query_ids:
                        raise ValueError(
                            "LightOn score rows must be grouped by query_id for memory-bounded artifact creation"
                        )
                    current_query_id = query_id
                positive_id = str(document_ids[0])
                current_positive_ids.add(positive_id)
                relevant_rows.append(
                    {
                        "query_id": query_id,
                        "document_id": positive_id,
                        "lightonai_positive_score": float(lightonai_scores[0]),
                    }
                )
                relevant_count += 1

                for retrieval_rank, (document_id, score) in enumerate(zip(document_ids[1:], lightonai_scores[1:])):
                    negative_id = str(document_id)
                    if negative_id == positive_id:
                        continue
                    score = float(score)
                    previous = current_candidates.get(negative_id)
                    if previous is None or score > float(previous["candidate_ranking"]):
                        current_candidates[negative_id] = {
                            "query_id": query_id,
                            "document_id": negative_id,
                            "candidate_ranking": score,
                            "lightonai_score": score,
                            "candidate_selected": True,
                            "retrieval_rank": int(retrieval_rank),
                            "retrieval_source": "lightonai",
                        }
                if len(relevant_rows) >= flush_rows:
                    flush_relevant()

        flush_candidate_query()
        flush_relevant()
        flush_candidates()
        relevant_writer.close()
        candidates_writer.close()
        os.replace(relevant_temp, relevant_path)
        os.replace(candidates_temp, candidates_path)
    except Exception:
        relevant_writer.close()
        candidates_writer.close()
        for temp_path in (relevant_temp, candidates_temp):
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise

    print(
        "Wrote LightOn pipeline artifacts to "
        f"{output_dir}: {query_count:,} queries, {corpus_count:,} documents, "
        f"{relevant_count:,} positives, {candidate_count:,} candidates"
    )


def export_lightonai_pipeline_artifacts(
    output_dir: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "fiqa",
    hf_cache_dir: str | None = None,
    load_num_proc: int | None = None,
) -> None:
    ensure_parent_dir(os.path.join(output_dir, "queries.parquet"))
    scores = _load_lightonai_component(dataset_name, "scores", split, hf_cache_dir, load_num_proc)
    queries = _load_lightonai_component(dataset_name, "queries", split, hf_cache_dir, load_num_proc)
    documents = _load_lightonai_component(dataset_name, "documents", split, hf_cache_dir, load_num_proc)
    write_lightonai_pipeline_artifacts(output_dir, queries, documents, scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LightOn splits into compact pipeline Parquet artifacts.")
    default_splits = config("LIGHTONAI_SPLITS", default=config("LIGHTONAI_SPLIT", default="fiqa"))
    parser.add_argument("--dataset_name", type=str, default=config("LIGHTONAI_DATASET_NAME", default=DEFAULT_DATASET_NAME))
    parser.add_argument("--split", type=str, default=first_split(default_splits))
    parser.add_argument("--output_root", type=str, default=config("LIGHTONAI_PIPELINE_ROOT", default="data/lightonai_pipeline"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit output dir. Defaults to LIGHTONAI_PIPELINE_ROOT/<split>.",
    )
    parser.add_argument("--hf_cache_dir", type=str, default=config("LIGHTONAI_HF_CACHE_DIR", default=None))
    parser.add_argument("--load_num_proc", type=int, default=_config_optional("LIGHTONAI_LOAD_NUM_PROC", cast=int))
    args = parser.parse_args()

    output_dir = args.output_dir or split_output_dir(args.output_root, args.split)
    export_lightonai_pipeline_artifacts(
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        hf_cache_dir=args.hf_cache_dir,
        load_num_proc=args.load_num_proc,
    )


if __name__ == "__main__":
    main()
