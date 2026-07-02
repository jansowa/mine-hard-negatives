from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from decouple import UndefinedValueError, config
from tqdm.auto import tqdm

from batch_tuning import (
    OOMRetryReranker,
    benchmark_reranker_batch_size,
    parse_batch_size_candidates,
    validate_batch_size_options,
)
from flag_embedding_jsonl import (
    as_text_list,
    count_complete_jsonl_rows,
    ensure_parent_dir,
    iter_jsonl_rows,
    validate_query_pos_neg_row,
)
from models import get_reranker_model, rerank

DEFAULT_POS_SCORE_FIELD = "pos_scores_stronger_reranker"
DEFAULT_NEG_SCORE_FIELD = "neg_scores_stronger_reranker"
DEFAULT_RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _config_optional(name: str, *, cast: Callable | None = None, default=None):
    try:
        value = config(name, default=None)
    except UndefinedValueError:
        return default
    if value in {None, ""}:
        return default
    if cast is None:
        return value
    return cast(value)


def _config_first(names: Sequence[str], *, cast: Callable | None = None, default=None):
    for name in names:
        value = _config_optional(name, cast=cast, default=None)
        if value is not None:
            return value
    return default


def incomplete_output_path(output_path: str) -> str:
    return f"{output_path}.incomplete"


def _write_jsonl_rows(handle, rows: Sequence[dict[str, Any]], fsync: bool) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    if fsync:
        os.fsync(handle.fileno())


def _score_list_with_missing(value: Any, field_name: str, expected_count: int) -> list[float | None]:
    if value is None:
        return [None] * expected_count
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if len(value) > expected_count:
        raise ValueError(f"{field_name} has {len(value)} values but expected at most {expected_count}")

    scores: list[float | None] = []
    for item in value:
        if item is None:
            scores.append(None)
        else:
            scores.append(float(item))
    scores.extend([None] * (expected_count - len(scores)))
    return scores


def _append_missing_pairs(
    *,
    query: str,
    docs: list[str],
    score_field: str,
    row: dict[str, Any],
    flat_queries: list[str],
    flat_docs: list[str],
    assignments: list[tuple[dict[str, Any], str, int]],
) -> None:
    scores = _score_list_with_missing(row.get(score_field), score_field, len(docs))
    row[score_field] = scores

    for index, (doc, score) in enumerate(zip(docs, scores)):
        if score is not None:
            continue
        flat_queries.append(query)
        flat_docs.append(doc)
        assignments.append((row, score_field, index))


def _score_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    reranker_model,
    safe_rerank: OOMRetryReranker,
    reranker_batch_size: int,
    pos_score_field: str,
    neg_score_field: str,
    score_negatives: bool,
) -> tuple[list[dict[str, Any]], int]:
    flat_queries: list[str] = []
    flat_docs: list[str] = []
    assignments: list[tuple[dict[str, Any], str, int]] = []

    for row in rows:
        validate_query_pos_neg_row(row)

        query = str(row["query"])
        pos = as_text_list(row.get("pos"), "pos")
        _append_missing_pairs(
            query=query,
            docs=pos,
            score_field=pos_score_field,
            row=row,
            flat_queries=flat_queries,
            flat_docs=flat_docs,
            assignments=assignments,
        )

        if score_negatives:
            neg = as_text_list(row.get("neg"), "neg")
            _append_missing_pairs(
                query=query,
                docs=neg,
                score_field=neg_score_field,
                row=row,
                flat_queries=flat_queries,
                flat_docs=flat_docs,
                assignments=assignments,
            )

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
    if len(scores) != len(assignments):
        raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(assignments)} pairs")

    for (row, score_field, index), score in zip(assignments, scores):
        row[score_field][index] = float(score)

    return rows, len(flat_docs)


def _collect_reranker_sample_pairs(
    input_path: str,
    skip_rows: int,
    sample_size: int,
    pos_score_field: str,
    neg_score_field: str,
    score_negatives: bool,
) -> tuple[list[str], list[str]]:
    if sample_size <= 0:
        return [], []

    sample_queries: list[str] = []
    sample_docs: list[str] = []
    for _, row in iter_jsonl_rows(input_path, skip_rows=skip_rows):
        validate_query_pos_neg_row(row)
        query = str(row["query"])

        pos = as_text_list(row.get("pos"), "pos")
        pos_scores = _score_list_with_missing(row.get(pos_score_field), pos_score_field, len(pos))
        for doc, score in zip(pos, pos_scores):
            if score is not None:
                continue
            sample_queries.append(query)
            sample_docs.append(doc)
            if len(sample_docs) >= sample_size:
                return sample_queries, sample_docs

        if not score_negatives:
            continue

        neg = as_text_list(row.get("neg"), "neg")
        neg_scores = _score_list_with_missing(row.get(neg_score_field), neg_score_field, len(neg))
        for doc, score in zip(neg, neg_scores):
            if score is not None:
                continue
            sample_queries.append(query)
            sample_docs.append(doc)
            if len(sample_docs) >= sample_size:
                return sample_queries, sample_docs

    return sample_queries, sample_docs


def add_stronger_reranker_scores_jsonl(
    input_path: str,
    output_path: str,
    reranker_model_name: str,
    reranker_batch_size: int | None = None,
    record_batch_size: int = 32,
    pos_score_field: str = DEFAULT_POS_SCORE_FIELD,
    neg_score_field: str = DEFAULT_NEG_SCORE_FIELD,
    score_negatives: bool = False,
    resume: bool = True,
    fsync: bool = False,
    report_path: str | None = None,
    auto_reranker_batch_size_candidates: str | None = None,
    auto_reranker_batch_size_min: int = 1,
    auto_reranker_batch_size_max: int = 64,
    auto_reranker_batch_size_sample_size: int = 128,
    auto_reranker_batch_size_memory_utilization: float = 0.70,
) -> dict[str, Any]:
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("input_path and output_path must be different; in-place updates are not supported")
    if not pos_score_field:
        raise ValueError("pos_score_field must not be empty")
    if score_negatives and not neg_score_field:
        raise ValueError("neg_score_field must not be empty when score_negatives is enabled")
    if record_batch_size <= 0:
        raise ValueError("record_batch_size must be greater than 0")

    validate_batch_size_options(
        reranker_batch_size,
        auto_reranker_batch_size_min,
        auto_reranker_batch_size_max,
        auto_reranker_batch_size_sample_size,
        auto_reranker_batch_size_memory_utilization,
        "reranker",
    )

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
    print("Stronger reranker loaded.")
    rerank_function = partial(rerank, model_name=reranker_model_name)

    if reranker_batch_size is None:
        candidates = parse_batch_size_candidates(
            auto_reranker_batch_size_candidates,
            minimum=auto_reranker_batch_size_min,
            maximum=auto_reranker_batch_size_max,
        )
        sample_queries, sample_docs = _collect_reranker_sample_pairs(
            input_path,
            processed_rows,
            auto_reranker_batch_size_sample_size,
            pos_score_field,
            neg_score_field,
            score_negatives,
        )
        reranker_batch_size = benchmark_reranker_batch_size(
            tokenizer,
            reranker_model,
            sample_queries,
            sample_docs,
            candidates,
            rerank_function,
            memory_utilization=auto_reranker_batch_size_memory_utilization,
        )
    else:
        print(f"Using explicit reranker batch size: {reranker_batch_size}")
    assert reranker_batch_size is not None

    safe_rerank = OOMRetryReranker(rerank_function, reranker_batch_size)
    print(f"Effective reranker batch size: {reranker_batch_size}")

    newly_scored_rows = 0
    newly_scored_pairs = 0
    batch: list[dict[str, Any]] = []

    with open(work_path, "a", encoding="utf-8") as output_handle:
        with tqdm(unit="row", desc="Adding stronger reranker JSONL scores", initial=processed_rows) as pbar:
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
                    pos_score_field,
                    neg_score_field,
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
                    pos_score_field,
                    neg_score_field,
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
        "pos_score_field": pos_score_field,
        "neg_score_field": neg_score_field,
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
    parser = argparse.ArgumentParser(description="Add stronger-reranker score fields to a FlagEmbedding-style JSONL.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default=_config_first(("FINAL_RERANKER_NAME", "RERANKER_NAME"), default=DEFAULT_RERANKER_NAME),
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=_config_first(("FINAL_RERANKER_BATCH_SIZE", "RERANKER_BATCH_SIZE"), cast=int),
        help="Explicit reranker batch size. If omitted, a short startup benchmark selects it automatically.",
    )
    parser.add_argument(
        "--record_batch_size",
        type=int,
        default=_config_first(
            ("STRONGER_RERANKER_JSONL_RECORD_BATCH_SIZE", "FLAG_JSONL_RECORD_BATCH_SIZE"),
            cast=int,
            default=32,
        ),
    )
    parser.add_argument("--pos_score_field", type=str, default=DEFAULT_POS_SCORE_FIELD)
    parser.add_argument("--neg_score_field", type=str, default=DEFAULT_NEG_SCORE_FIELD)
    parser.add_argument("--score_negatives", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument(
        "--auto_reranker_batch_size_candidates",
        type=str,
        default=_config_optional("FINAL_AUTO_RERANKER_BATCH_SIZE_CANDIDATES"),
        help="Comma-separated reranker batch sizes to benchmark. Defaults to powers of two between min and max.",
    )
    parser.add_argument(
        "--auto_reranker_batch_size_min",
        type=int,
        default=_config_optional("FINAL_AUTO_RERANKER_BATCH_SIZE_MIN", cast=int, default=1),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_max",
        type=int,
        default=_config_optional("FINAL_AUTO_RERANKER_BATCH_SIZE_MAX", cast=int, default=64),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_sample_size",
        type=int,
        default=_config_optional("FINAL_AUTO_RERANKER_BATCH_SIZE_SAMPLE_SIZE", cast=int, default=128),
    )
    parser.add_argument(
        "--auto_reranker_batch_size_memory_utilization",
        type=float,
        default=_config_optional("FINAL_AUTO_RERANKER_BATCH_SIZE_MEMORY_UTILIZATION", cast=float, default=0.70),
    )
    args = parser.parse_args()

    report_path = args.report_path or f"{args.output_path}.report.json"
    add_stronger_reranker_scores_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        reranker_model_name=args.reranker_model_name,
        reranker_batch_size=args.reranker_batch_size,
        record_batch_size=args.record_batch_size,
        pos_score_field=args.pos_score_field,
        neg_score_field=args.neg_score_field,
        score_negatives=args.score_negatives,
        resume=args.resume,
        fsync=args.fsync,
        report_path=report_path,
        auto_reranker_batch_size_candidates=args.auto_reranker_batch_size_candidates,
        auto_reranker_batch_size_min=args.auto_reranker_batch_size_min,
        auto_reranker_batch_size_max=args.auto_reranker_batch_size_max,
        auto_reranker_batch_size_sample_size=args.auto_reranker_batch_size_sample_size,
        auto_reranker_batch_size_memory_utilization=args.auto_reranker_batch_size_memory_utilization,
    )


if __name__ == "__main__":
    main()
