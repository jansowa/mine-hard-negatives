from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from decouple import UndefinedValueError, config
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from curated_negatives.flag_embedding import (
        as_text_list,
        count_complete_jsonl_rows,
        ensure_parent_dir,
        iter_jsonl_rows,
        move_existing_scores,
        validate_flag_embedding_row,
    )
else:
    from .flag_embedding import (
        as_text_list,
        count_complete_jsonl_rows,
        ensure_parent_dir,
        iter_jsonl_rows,
        move_existing_scores,
        validate_flag_embedding_row,
    )

from batch_tuning import OOMRetryReranker
from models import get_reranker_model, rerank


def _config_optional(name: str, *, cast: Callable | None = None, default=None):
    try:
        if cast is None:
            return config(name, default=default)
        return config(name, cast=cast, default=default)
    except UndefinedValueError:
        return default


def incomplete_output_path(output_path: str) -> str:
    return f"{output_path}.incomplete"


def _write_jsonl_rows(handle, rows: Sequence[dict[str, Any]], fsync: bool) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    if fsync:
        os.fsync(handle.fileno())


def _score_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    reranker_model,
    safe_rerank: OOMRetryReranker,
    reranker_batch_size: int,
    backup_score_prefix: str,
    score_positives: bool,
    score_negatives: bool,
) -> tuple[list[dict[str, Any]], int]:
    flat_queries: list[str] = []
    flat_docs: list[str] = []
    spans: list[tuple[int | None, int | None]] = []

    for row in rows:
        validate_flag_embedding_row(row)
        move_existing_scores(row, backup_score_prefix)

        query = str(row["query"])
        pos = as_text_list(row.get("pos"), "pos")
        neg = as_text_list(row.get("neg"), "neg")

        pos_start = None
        if score_positives:
            pos_start = len(flat_docs)
            flat_queries.extend([query] * len(pos))
            flat_docs.extend(pos)

        neg_start = None
        if score_negatives:
            neg_start = len(flat_docs)
            flat_queries.extend([query] * len(neg))
            flat_docs.extend(neg)

        spans.append((pos_start, neg_start))

    if flat_docs:
        scores = safe_rerank(
            tokenizer,
            reranker_model,
            flat_queries,
            flat_docs,
            batch_size=reranker_batch_size,
        )
    else:
        scores = []

    for row, (pos_start, neg_start) in zip(rows, spans):
        pos_count = len(as_text_list(row.get("pos"), "pos"))
        neg_count = len(as_text_list(row.get("neg"), "neg"))
        if score_positives:
            assert pos_start is not None
            row["pos_scores"] = [float(score) for score in scores[pos_start : pos_start + pos_count]]
        if score_negatives:
            assert neg_start is not None
            row["neg_scores"] = [float(score) for score in scores[neg_start : neg_start + neg_count]]

    return rows, len(flat_docs)


def score_flag_embedding_jsonl(
    input_path: str,
    output_path: str,
    reranker_model_name: str,
    reranker_batch_size: int = 16,
    record_batch_size: int = 32,
    backup_score_prefix: str = "original_",
    score_positives: bool = True,
    score_negatives: bool = True,
    resume: bool = True,
    fsync: bool = False,
    report_path: str | None = None,
) -> dict[str, Any]:
    if reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size must be greater than 0")
    if record_batch_size <= 0:
        raise ValueError("record_batch_size must be greater than 0")
    if not score_positives and not score_negatives:
        raise ValueError("At least one of score_positives or score_negatives must be enabled")

    ensure_parent_dir(output_path)
    work_path = incomplete_output_path(output_path)

    if resume and not os.path.exists(work_path) and os.path.exists(output_path):
        completed_rows = count_complete_jsonl_rows(output_path)
        print(f"{output_path} already exists with {completed_rows:,} rows; nothing to resume.")
        return {
            "completed": True,
            "input_path": input_path,
            "output_path": output_path,
            "already_completed_rows": completed_rows,
            "newly_scored_rows": 0,
            "newly_scored_pairs": 0,
        }

    if not resume and os.path.exists(work_path):
        os.remove(work_path)

    processed_rows = count_complete_jsonl_rows(work_path, truncate_invalid_tail=True) if resume else 0
    if processed_rows:
        print(f"Resume enabled: found {processed_rows:,} already scored rows in {work_path}")

    tokenizer, reranker_model = get_reranker_model(reranker_model_name)
    print("Reranker loaded.")
    rerank_function = partial(rerank, model_name=reranker_model_name)
    safe_rerank = OOMRetryReranker(rerank_function, reranker_batch_size)

    newly_scored_rows = 0
    newly_scored_pairs = 0
    batch: list[dict[str, Any]] = []

    with open(work_path, "a", encoding="utf-8") as output_handle:
        with tqdm(unit="row", desc="Scoring FlagEmbedding JSONL", initial=processed_rows) as pbar:
            for _, row in iter_jsonl_rows(input_path, skip_rows=processed_rows):
                batch.append(row)
                if len(batch) < record_batch_size:
                    continue

                scored_rows, pair_count = _score_rows(
                    batch,
                    tokenizer,
                    reranker_model,
                    safe_rerank,
                    reranker_batch_size,
                    backup_score_prefix,
                    score_positives,
                    score_negatives,
                )
                _write_jsonl_rows(output_handle, scored_rows, fsync)
                newly_scored_rows += len(scored_rows)
                newly_scored_pairs += pair_count
                pbar.update(len(scored_rows))
                batch = []

            if batch:
                scored_rows, pair_count = _score_rows(
                    batch,
                    tokenizer,
                    reranker_model,
                    safe_rerank,
                    reranker_batch_size,
                    backup_score_prefix,
                    score_positives,
                    score_negatives,
                )
                _write_jsonl_rows(output_handle, scored_rows, fsync)
                newly_scored_rows += len(scored_rows)
                newly_scored_pairs += pair_count
                pbar.update(len(scored_rows))

    os.replace(work_path, output_path)
    report = {
        "completed": True,
        "input_path": input_path,
        "output_path": output_path,
        "reranker_model_name": reranker_model_name,
        "reranker_batch_size": reranker_batch_size,
        "record_batch_size": record_batch_size,
        "backup_score_prefix": backup_score_prefix,
        "score_positives": score_positives,
        "score_negatives": score_negatives,
        "resumed_rows": processed_rows,
        "newly_scored_rows": newly_scored_rows,
        "newly_scored_pairs": newly_scored_pairs,
        "total_output_rows": processed_rows + newly_scored_rows,
    }
    if report_path:
        ensure_parent_dir(report_path)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore a FlagEmbedding JSONL dataset with a reranker.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default=_config_optional(
            "FINAL_RERANKER_NAME",
            default=_config_optional("RERANKER_NAME", default="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ),
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=_config_optional(
            "FINAL_RERANKER_BATCH_SIZE",
            cast=int,
            default=_config_optional("RERANKER_BATCH_SIZE", cast=int, default=16),
        ),
    )
    parser.add_argument(
        "--record_batch_size",
        type=int,
        default=_config_optional("FLAG_JSONL_RECORD_BATCH_SIZE", cast=int, default=32),
    )
    parser.add_argument("--backup_score_prefix", type=str, default="original_")
    parser.add_argument("--score_positives", dest="score_positives", action="store_true")
    parser.add_argument("--no_score_positives", dest="score_positives", action="store_false")
    parser.set_defaults(score_positives=True)
    parser.add_argument("--score_negatives", dest="score_negatives", action="store_true")
    parser.add_argument("--no_score_negatives", dest="score_negatives", action="store_false")
    parser.set_defaults(score_negatives=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument("--report_path", type=str, default=None)
    args = parser.parse_args()

    report_path = args.report_path or f"{args.output_path}.report.json"
    score_flag_embedding_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        reranker_model_name=args.reranker_model_name,
        reranker_batch_size=args.reranker_batch_size,
        record_batch_size=args.record_batch_size,
        backup_score_prefix=args.backup_score_prefix,
        score_positives=args.score_positives,
        score_negatives=args.score_negatives,
        resume=args.resume,
        fsync=args.fsync,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()

