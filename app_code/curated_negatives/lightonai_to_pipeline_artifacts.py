from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import datasets
import pandas as pd
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
    query_frame, corpus_frame, relevant_frame, candidates_frame = build_lightonai_pipeline_artifact_frames(
        queries,
        documents,
        scores,
    )
    os.makedirs(output_dir, exist_ok=True)
    query_frame.to_parquet(os.path.join(output_dir, "queries.parquet"), index=False)
    corpus_frame.to_parquet(os.path.join(output_dir, "corpus.parquet"), index=False)
    relevant_frame.to_parquet(os.path.join(output_dir, "relevant.parquet"), index=False)
    candidates_frame.to_parquet(os.path.join(output_dir, "negative_candidates.parquet"), index=False)
    print(
        "Wrote LightOn pipeline artifacts to "
        f"{output_dir}: {len(query_frame):,} queries, {len(corpus_frame):,} documents, "
        f"{len(relevant_frame):,} positives, {len(candidates_frame):,} candidates"
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
