from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from decouple import UndefinedValueError, config

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from curated_negatives.lightonai_to_flag_embedding import DEFAULT_DATASET_NAME
    from curated_negatives.lightonai_to_pipeline_artifacts import (
        export_lightonai_pipeline_artifacts,
        parse_splits,
        split_output_dir,
    )
else:
    from .lightonai_to_flag_embedding import DEFAULT_DATASET_NAME
    from .lightonai_to_pipeline_artifacts import export_lightonai_pipeline_artifacts, parse_splits, split_output_dir

from add_positives_ranks import get_positive_ranks_auto_batch_defaults, process_relevant
from create_flag_embedding_jsonl import process_negatives_streaming
from rerank_negative_candidates import rerank_candidates

STAGE_ORDER = ("artifacts", "positives", "negatives", "jsonl")


@dataclass(frozen=True)
class SplitPaths:
    output_dir: str
    queries_path: str
    corpus_path: str
    relevant_path: str
    relevant_with_score_path: str
    negative_candidates_path: str
    negatives_path: str
    output_jsonl_path: str
    corpus_sqlite_path: str
    negcount_sqlite_path: str
    negatives_report_path: str
    jsonl_report_path: str


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


def _config_first(names: tuple[str, ...], *, cast=None, default=None):
    for name in names:
        value = _config_optional(name, cast=cast, default=None)
        if value is not None:
            return value
    return default


def _parse_stages(raw_stages: str) -> list[str]:
    stages = [stage.strip() for stage in raw_stages.split(",") if stage.strip()]
    if not stages:
        raise ValueError("At least one stage is required")
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown LightOn pipeline stages: {', '.join(unknown)}")
    return [stage for stage in STAGE_ORDER if stage in stages]


def split_paths(output_root: str, split: str) -> SplitPaths:
    output_dir = split_output_dir(output_root, split)
    return SplitPaths(
        output_dir=output_dir,
        queries_path=os.path.join(output_dir, "queries.parquet"),
        corpus_path=os.path.join(output_dir, "corpus.parquet"),
        relevant_path=os.path.join(output_dir, "relevant.parquet"),
        relevant_with_score_path=os.path.join(output_dir, "relevant_with_score.parquet"),
        negative_candidates_path=os.path.join(output_dir, "negative_candidates.parquet"),
        negatives_path=os.path.join(output_dir, "negatives.parquet"),
        output_jsonl_path=os.path.join(output_dir, "train.jsonl"),
        corpus_sqlite_path=os.path.join(output_dir, "corpus.sqlite"),
        negcount_sqlite_path=os.path.join(output_dir, "negcount.sqlite"),
        negatives_report_path=os.path.join(output_dir, "negatives.parquet.report.json"),
        jsonl_report_path=os.path.join(output_dir, "train.jsonl.report.json"),
    )


def run_lightonai_split(
    split: str,
    output_root: str,
    dataset_name: str,
    stages: list[str],
    hf_cache_dir: str | None,
    load_num_proc: int | None,
) -> None:
    paths = split_paths(output_root, split)
    print(f"\n=== LightOn split: {split} -> {paths.output_dir} ===")

    if "artifacts" in stages:
        export_lightonai_pipeline_artifacts(
            output_dir=paths.output_dir,
            dataset_name=dataset_name,
            split=split,
            hf_cache_dir=hf_cache_dir,
            load_num_proc=load_num_proc,
        )

    if "positives" in stages:
        auto_defaults = get_positive_ranks_auto_batch_defaults(final_step=True)
        process_relevant(
            queries_path=paths.queries_path,
            corpus_path=paths.corpus_path,
            relevant_path=paths.relevant_path,
            output_path=paths.relevant_with_score_path,
            chunk_size=config("PROCESSING_CHUNK_SIZE", cast=int, default=100_000),
            reranker_batch_size=_config_optional("FINAL_RERANKER_BATCH_SIZE", cast=int),
            reranker_model_name=_config_first(
                ("FINAL_RERANKER_NAME", "FINAL_POSITIVE_RANKS_RERANKER_NAME", "RERANKER_NAME"),
                default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            score_column=_config_first(
                ("FINAL_POSITIVE_RANKS_SCORE_COLUMN", "FINAL_POSITIVE_SCORE_COLUMN", "POSITIVE_SCORE_COLUMN"),
                default="positive_ranking",
            ),
            auto_reranker_batch_size_candidates=auto_defaults["candidates"],
            auto_reranker_batch_size_min=auto_defaults["min"],
            auto_reranker_batch_size_max=auto_defaults["max"],
            auto_reranker_batch_size_sample_size=auto_defaults["sample_size"],
            auto_reranker_batch_size_memory_utilization=auto_defaults["memory_utilization"],
        )

    if "negatives" in stages:
        num_negatives = config("NUM_NEGATIVES", cast=int, default=10)
        rerank_candidates(
            candidates_path=paths.negative_candidates_path,
            queries_path=paths.queries_path,
            corpus_path=paths.corpus_path,
            relevant_path=paths.relevant_with_score_path,
            output_path=paths.negatives_path,
            reranker_model_name=_config_first(
                ("FINAL_RERANKER_NAME", "RERANKER_NAME"),
                default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            reranker_batch_size=_config_optional("FINAL_RERANKER_BATCH_SIZE", cast=int),
            ranking_column=config("FINAL_RANKING_COLUMN", default="final_ranking"),
            candidate_selected_column=config("CANDIDATE_SELECTED_COLUMN", default="candidate_selected"),
            selected_only=config("FINAL_RERANK_SELECTED_ONLY", cast=bool, default=True),
            chunk_size=config("PROCESSING_CHUNK_SIZE", cast=int, default=100_000),
            resume=config("FINAL_RERANK_RESUME", cast=bool, default=True),
            row_group_size=config("NEGATIVES_PARQUET_ROW_GROUP_SIZE", cast=int, default=100_000),
            rerank_mode=config("FINAL_RERANK_MODE", default="adaptive"),
            candidate_score_column=config("NEGATIVE_RANKING_COLUMN", default="candidate_ranking"),
            positive_score_column=config("FINAL_POSITIVE_SCORE_COLUMN", default="positive_ranking"),
            num_negatives=num_negatives,
            beta=config("FINAL_BETA", cast=float, default=config("BETA", cast=float, default=0.01)),
            u_floor=config("FINAL_U_FLOOR", cast=float, default=config("U_FLOOR", cast=float, default=0.005)),
            initial_budget=config("FINAL_RERANK_INITIAL_BUDGET", cast=int, default=max(1, num_negatives * 2)),
            budget_step=config("FINAL_RERANK_BUDGET_STEP", cast=int, default=max(1, num_negatives)),
            max_budget=config("FINAL_RERANK_MAX_BUDGET", cast=int, default=max(1, num_negatives * 8)),
            report_path=paths.negatives_report_path,
            auto_reranker_batch_size_candidates=config("FINAL_AUTO_RERANKER_BATCH_SIZE_CANDIDATES", default=None),
            auto_reranker_batch_size_min=config("FINAL_AUTO_RERANKER_BATCH_SIZE_MIN", cast=int, default=1),
            auto_reranker_batch_size_max=config("FINAL_AUTO_RERANKER_BATCH_SIZE_MAX", cast=int, default=64),
            auto_reranker_batch_size_sample_size=config("FINAL_AUTO_RERANKER_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=128),
            auto_reranker_batch_size_memory_utilization=config(
                "FINAL_AUTO_RERANKER_BATCH_SIZE_MEMORY_UTILIZATION",
                cast=float,
                default=0.70,
            ),
        )

    if "jsonl" in stages:
        process_negatives_streaming(
            corpus_path=paths.corpus_path,
            queries_path=paths.queries_path,
            relevant_path=paths.relevant_with_score_path,
            negatives_path=paths.negatives_path,
            output_path=paths.output_jsonl_path,
            num_negatives=config("NUM_NEGATIVES", cast=int, default=10),
            beta=config("FINAL_BETA", cast=float, default=config("BETA", cast=float, default=0.01)),
            u_floor=config("FINAL_U_FLOOR", cast=float, default=config("U_FLOOR", cast=float, default=0.005)),
            max_neg_reuse=config("MAX_NEG_REUSE", cast=int, default=1000),
            corpus_sqlite_path=paths.corpus_sqlite_path,
            negcount_sqlite_path=paths.negcount_sqlite_path,
            query_chunk_size=config("QUERY_CHUNK_SIZE", cast=int, default=10_000),
            oversample_factor=config("OVERSAMPLE_FACTOR", cast=int, default=5),
            positive_score_column=config("POSITIVE_SCORE_COLUMN", default="positive_ranking"),
            negative_score_column=config("NEGATIVE_SCORE_COLUMN", default="final_ranking"),
            positive_original_score_column=config(
                "POSITIVE_ORIGINAL_SCORE_COLUMN",
                default="lightonai_positive_score",
            ),
            negative_original_score_column=config("NEGATIVE_ORIGINAL_SCORE_COLUMN", default="candidate_ranking"),
            prompt=config("FLAG_EMBEDDING_PROMPT", default=""),
            dataset_type=config("FLAG_EMBEDDING_TYPE", default="retrieval"),
            backfill_policy=config("BACKFILL_POLICY", default="relaxed"),
            report_path=paths.jsonl_report_path,
        )


def run_lightonai_pipeline(
    splits: list[str],
    output_root: str,
    dataset_name: str,
    stages: list[str],
    hf_cache_dir: str | None = None,
    load_num_proc: int | None = None,
) -> None:
    for split in splits:
        run_lightonai_split(
            split=split,
            output_root=output_root,
            dataset_name=dataset_name,
            stages=stages,
            hf_cache_dir=hf_cache_dir,
            load_num_proc=load_num_proc,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the env-driven LightOn adaptive compact pipeline.")
    parser.add_argument("--splits", type=str, default=config("LIGHTONAI_SPLITS", default="fiqa"))
    parser.add_argument("--dataset_name", type=str, default=config("LIGHTONAI_DATASET_NAME", default=DEFAULT_DATASET_NAME))
    parser.add_argument("--output_root", type=str, default=config("LIGHTONAI_PIPELINE_ROOT", default="data/lightonai_pipeline"))
    parser.add_argument(
        "--stages",
        type=str,
        default=config("LIGHTONAI_STAGES", default="artifacts,positives,negatives,jsonl"),
        help=f"Comma-separated subset of: {', '.join(STAGE_ORDER)}",
    )
    parser.add_argument("--hf_cache_dir", type=str, default=config("LIGHTONAI_HF_CACHE_DIR", default=None))
    parser.add_argument("--load_num_proc", type=int, default=_config_optional("LIGHTONAI_LOAD_NUM_PROC", cast=int))
    args = parser.parse_args()

    run_lightonai_pipeline(
        splits=parse_splits(args.splits),
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        stages=_parse_stages(args.stages),
        hf_cache_dir=args.hf_cache_dir,
        load_num_proc=args.load_num_proc,
    )


if __name__ == "__main__":
    main()
