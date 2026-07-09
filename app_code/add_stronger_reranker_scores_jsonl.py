from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
DEFAULT_PARQUET_SCORE_FIELD = "score_stronger_reranker"
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


def _is_parquet_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".parquet"


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


def _parquet_output_schema(input_schema: pa.Schema, score_column: str) -> pa.Schema:
    fields = [field for field in input_schema if field.name != score_column]
    fields.append(pa.field(score_column, pa.float32()))
    return pa.schema(fields)


def parquet_resume_source_path(output_path: str) -> str:
    return f"{incomplete_output_path(output_path)}.resume_source"


def _iter_parquet_batches_from_row(
    parquet_file: pq.ParquetFile,
    *,
    batch_size: int,
    skip_rows: int = 0,
):
    seen_rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_rows = batch.num_rows
        batch_start = seen_rows
        batch_end = seen_rows + batch_rows
        seen_rows = batch_end

        if batch_end <= skip_rows:
            continue

        local_start = max(0, skip_rows - batch_start)
        yield batch.slice(local_start, batch_rows - local_start)


def _open_parquet_resume_source(path: str, output_schema: pa.Schema) -> pq.ParquetFile:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume from {path}; the incomplete parquet is not readable. "
            "If the previous process was killed before pyarrow wrote the footer, the already-scored rows cannot be "
            "recovered from this file."
        ) from exc

    missing_columns = [name for name in output_schema.names if name not in parquet_file.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"Cannot resume from {path}; missing column(s): {', '.join(missing_columns)}")
    return parquet_file


def _close_parquet_file(parquet_file: pq.ParquetFile) -> None:
    close = getattr(parquet_file, "close", None)
    if close is not None:
        close()


def _copy_parquet_resume_rows(
    *,
    resume_path: str,
    writer: pq.ParquetWriter,
    output_schema: pa.Schema,
    batch_size: int,
) -> int:
    parquet_file = _open_parquet_resume_source(resume_path, output_schema)
    copied_rows = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch]).select(output_schema.names)
            if table.schema != output_schema:
                table = table.cast(output_schema, safe=False)
            writer.write_table(table)
            copied_rows += table.num_rows
        return copied_rows
    finally:
        _close_parquet_file(parquet_file)


def _validate_parquet_columns(
    input_schema: pa.Schema,
    *,
    question_column: str,
    answer_column: str,
    score_column: str,
    rejected_column: str,
    only_verified: bool,
) -> None:
    if not question_column:
        raise ValueError("question_column must not be empty")
    if not answer_column:
        raise ValueError("answer_column must not be empty")
    if not score_column:
        raise ValueError("parquet_score_column must not be empty")
    if only_verified and not rejected_column:
        raise ValueError("rejected_column must not be empty when only_verified is enabled")

    column_names = set(input_schema.names)
    missing = [name for name in (question_column, answer_column) if name not in column_names]
    if missing:
        raise ValueError(f"Parquet input is missing required column(s): {', '.join(missing)}")
    if only_verified and rejected_column not in column_names:
        raise ValueError(f"Parquet input is missing required column {rejected_column!r} for --only-verified")


def _existing_parquet_scores(df: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float32")
    return pd.to_numeric(df[score_column], errors="coerce")


def _verified_parquet_rows(df: pd.DataFrame, rejected_column: str) -> pd.Series:
    return df[rejected_column].eq(False).fillna(False)


def _parquet_rows_to_score(
    df: pd.DataFrame,
    *,
    question_column: str,
    answer_column: str,
    score_column: str,
    rejected_column: str,
    only_verified: bool,
) -> tuple[pd.Series, pd.Series]:
    score_values = _existing_parquet_scores(df, score_column)
    mask = score_values.isna() & df[question_column].notna() & df[answer_column].notna()
    if only_verified:
        mask &= _verified_parquet_rows(df, rejected_column)
    return score_values, mask


def _score_parquet_batch(
    batch: pa.RecordBatch,
    *,
    tokenizer,
    reranker_model,
    safe_rerank: OOMRetryReranker,
    reranker_batch_size: int,
    output_schema: pa.Schema,
    question_column: str,
    answer_column: str,
    score_column: str,
    rejected_column: str,
    only_verified: bool,
) -> tuple[pa.Table, int]:
    df = batch.to_pandas()
    score_values, mask = _parquet_rows_to_score(
        df,
        question_column=question_column,
        answer_column=answer_column,
        score_column=score_column,
        rejected_column=rejected_column,
        only_verified=only_verified,
    )

    scored_count = int(mask.sum())
    if scored_count:
        score_rows = df.loc[mask, [question_column, answer_column]]
        queries = score_rows[question_column].astype(str).tolist()
        answers = score_rows[answer_column].astype(str).tolist()
        scores = safe_rerank(
            tokenizer,
            reranker_model,
            queries,
            answers,
            batch_size=reranker_batch_size,
        )
        if len(scores) != scored_count:
            raise RuntimeError(f"Reranker returned {len(scores)} scores for {scored_count} pairs")
        score_values.loc[mask] = [float(score) for score in scores]

    result_df = df.drop(columns=[score_column], errors="ignore")
    result_df[score_column] = score_values.astype("float32")
    result_df = result_df[list(output_schema.names)]
    return pa.Table.from_pandas(result_df, schema=output_schema, preserve_index=False), scored_count


def _collect_parquet_reranker_sample_pairs(
    input_path: str,
    sample_size: int,
    *,
    skip_rows: int = 0,
    question_column: str,
    answer_column: str,
    score_column: str,
    rejected_column: str,
    only_verified: bool,
) -> tuple[list[str], list[str]]:
    if sample_size <= 0:
        return [], []

    parquet_file = pq.ParquetFile(input_path)
    schema_names = set(parquet_file.schema_arrow.names)
    columns = [question_column, answer_column]
    if score_column in schema_names:
        columns.append(score_column)
    if only_verified:
        columns.append(rejected_column)

    sample_queries: list[str] = []
    sample_docs: list[str] = []
    batch_size = max(1, min(65_536, max(sample_size * 4, 1_024)))
    for batch in _iter_parquet_batches_from_row(parquet_file, batch_size=batch_size, skip_rows=skip_rows):
        df = pa.Table.from_batches([batch]).select(columns).to_pandas()
        _, mask = _parquet_rows_to_score(
            df,
            question_column=question_column,
            answer_column=answer_column,
            score_column=score_column,
            rejected_column=rejected_column,
            only_verified=only_verified,
        )
        if not mask.any():
            continue

        for query, answer in df.loc[mask, [question_column, answer_column]].itertuples(index=False, name=None):
            sample_queries.append(str(query))
            sample_docs.append(str(answer))
            if len(sample_docs) >= sample_size:
                return sample_queries, sample_docs

    return sample_queries, sample_docs


def add_stronger_reranker_scores_parquet(
    input_path: str,
    output_path: str,
    reranker_model_name: str,
    reranker_batch_size: int | None = None,
    record_batch_size: int = 32,
    score_column: str = DEFAULT_PARQUET_SCORE_FIELD,
    question_column: str = "question",
    answer_column: str = "answer",
    rejected_column: str = "rejected",
    only_verified: bool = False,
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
    if not _is_parquet_path(output_path):
        raise ValueError("Parquet input requires output_path ending with .parquet")
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
    resume_path = parquet_resume_source_path(output_path)
    input_parquet = pq.ParquetFile(input_path)
    input_schema = input_parquet.schema_arrow
    total_rows = input_parquet.metadata.num_rows
    _validate_parquet_columns(
        input_schema,
        question_column=question_column,
        answer_column=answer_column,
        score_column=score_column,
        rejected_column=rejected_column,
        only_verified=only_verified,
    )
    output_schema = _parquet_output_schema(input_schema, score_column)

    if resume and not os.path.exists(work_path) and os.path.exists(output_path):
        completed_rows = pq.ParquetFile(output_path).metadata.num_rows
        print(f"{output_path} already exists with {completed_rows:,} rows; nothing to resume.")
        return {
            "completed": True,
            "input_path": input_path,
            "output_path": output_path,
            "format": "parquet",
            "already_completed_rows": completed_rows,
            "newly_scored_rows": 0,
            "newly_scored_pairs": 0,
        }

    resumed_rows = 0
    resume_source_path: str | None = None
    if resume:
        resume_candidates: list[tuple[int, str]] = []
        resume_errors: list[Exception] = []
        for candidate_path in (work_path, resume_path):
            if not os.path.exists(candidate_path):
                continue
            try:
                candidate_file = _open_parquet_resume_source(candidate_path, output_schema)
                try:
                    resume_candidates.append((candidate_file.metadata.num_rows, candidate_path))
                finally:
                    _close_parquet_file(candidate_file)
            except Exception as exc:
                resume_errors.append(exc)

        if resume_candidates:
            resumed_rows, selected_resume_path = max(resume_candidates, key=lambda item: item[0])
            if resumed_rows > total_rows:
                raise ValueError(
                    f"Cannot resume from {selected_resume_path}; it has {resumed_rows:,} rows but input has "
                    f"{total_rows:,} rows"
                )
            if resumed_rows:
                if selected_resume_path != resume_path:
                    if os.path.exists(resume_path):
                        os.remove(resume_path)
                    os.replace(selected_resume_path, resume_path)
                elif os.path.exists(work_path):
                    os.remove(work_path)
                resume_source_path = resume_path
                print(f"Resume enabled: found {resumed_rows:,} already scored parquet rows in {resume_source_path}")
            else:
                if os.path.exists(work_path):
                    os.remove(work_path)
                if os.path.exists(resume_path):
                    os.remove(resume_path)
        elif resume_errors:
            raise resume_errors[0]
    else:
        if os.path.exists(work_path):
            os.remove(work_path)
        if os.path.exists(resume_path):
            os.remove(resume_path)

    if resumed_rows == total_rows and resume_source_path is not None:
        writer = pq.ParquetWriter(work_path, output_schema, compression="zstd", use_dictionary=True)
        try:
            copied_rows = _copy_parquet_resume_rows(
                resume_path=resume_source_path,
                writer=writer,
                output_schema=output_schema,
                batch_size=record_batch_size,
            )
            if copied_rows != resumed_rows:
                raise RuntimeError(f"Copied {copied_rows:,} resume rows but expected {resumed_rows:,}")
        finally:
            writer.close()

        if fsync:
            with open(work_path, "r+b") as handle:
                os.fsync(handle.fileno())
        os.replace(work_path, output_path)
        if os.path.exists(resume_source_path):
            os.remove(resume_source_path)
        report = {
            "completed": True,
            "input_path": input_path,
            "output_path": output_path,
            "format": "parquet",
            "reranker_model_name": reranker_model_name,
            "reranker_batch_size": reranker_batch_size,
            "record_batch_size": record_batch_size,
            "parquet_score_column": score_column,
            "question_column": question_column,
            "answer_column": answer_column,
            "rejected_column": rejected_column,
            "only_verified": only_verified,
            "resumed_rows": resumed_rows,
            "newly_scored_rows": 0,
            "newly_scored_pairs": 0,
            "total_output_rows": resumed_rows,
        }
        if report_path:
            ensure_parent_dir(report_path)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        return report

    tokenizer, reranker_model = get_reranker_model(reranker_model_name)
    print("Stronger reranker loaded.")
    rerank_function = partial(rerank, model_name=reranker_model_name)

    if reranker_batch_size is None:
        candidates = parse_batch_size_candidates(
            auto_reranker_batch_size_candidates,
            minimum=auto_reranker_batch_size_min,
            maximum=auto_reranker_batch_size_max,
        )
        sample_queries, sample_docs = _collect_parquet_reranker_sample_pairs(
            input_path,
            auto_reranker_batch_size_sample_size,
            skip_rows=resumed_rows,
            question_column=question_column,
            answer_column=answer_column,
            score_column=score_column,
            rejected_column=rejected_column,
            only_verified=only_verified,
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
    total_output_rows = 0
    writer = pq.ParquetWriter(work_path, output_schema, compression="zstd", use_dictionary=True)
    try:
        if resume_source_path is not None:
            copied_rows = _copy_parquet_resume_rows(
                resume_path=resume_source_path,
                writer=writer,
                output_schema=output_schema,
                batch_size=record_batch_size,
            )
            if copied_rows != resumed_rows:
                raise RuntimeError(f"Copied {copied_rows:,} resume rows but expected {resumed_rows:,}")
            total_output_rows += copied_rows

        with tqdm(
            total=total_rows,
            unit="row",
            desc="Adding stronger reranker parquet scores",
            initial=resumed_rows,
        ) as pbar:
            for batch in _iter_parquet_batches_from_row(
                input_parquet,
                batch_size=record_batch_size,
                skip_rows=resumed_rows,
            ):
                output_table, scored_count = _score_parquet_batch(
                    batch,
                    tokenizer=tokenizer,
                    reranker_model=reranker_model,
                    safe_rerank=safe_rerank,
                    reranker_batch_size=reranker_batch_size,
                    output_schema=output_schema,
                    question_column=question_column,
                    answer_column=answer_column,
                    score_column=score_column,
                    rejected_column=rejected_column,
                    only_verified=only_verified,
                )
                writer.write_table(output_table)
                newly_scored_rows += scored_count
                total_output_rows += output_table.num_rows
                pbar.update(output_table.num_rows)
    finally:
        writer.close()

    if fsync:
        with open(work_path, "r+b") as handle:
            os.fsync(handle.fileno())

    os.replace(work_path, output_path)
    if resume_source_path is not None and os.path.exists(resume_source_path):
        os.remove(resume_source_path)
    report = {
        "completed": True,
        "input_path": input_path,
        "output_path": output_path,
        "format": "parquet",
        "reranker_model_name": reranker_model_name,
        "reranker_batch_size": reranker_batch_size,
        "record_batch_size": record_batch_size,
        "parquet_score_column": score_column,
        "question_column": question_column,
        "answer_column": answer_column,
        "rejected_column": rejected_column,
        "only_verified": only_verified,
        "resumed_rows": resumed_rows,
        "newly_scored_rows": newly_scored_rows,
        "newly_scored_pairs": newly_scored_rows,
        "total_output_rows": total_output_rows,
    }
    if report_path:
        ensure_parent_dir(report_path)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    return report


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
    parquet_score_column: str = DEFAULT_PARQUET_SCORE_FIELD,
    question_column: str = "question",
    answer_column: str = "answer",
    rejected_column: str = "rejected",
    only_verified: bool = False,
) -> dict[str, Any]:
    if _is_parquet_path(input_path):
        return add_stronger_reranker_scores_parquet(
            input_path=input_path,
            output_path=output_path,
            reranker_model_name=reranker_model_name,
            reranker_batch_size=reranker_batch_size,
            record_batch_size=record_batch_size,
            score_column=parquet_score_column,
            question_column=question_column,
            answer_column=answer_column,
            rejected_column=rejected_column,
            only_verified=only_verified,
            resume=resume,
            fsync=fsync,
            report_path=report_path,
            auto_reranker_batch_size_candidates=auto_reranker_batch_size_candidates,
            auto_reranker_batch_size_min=auto_reranker_batch_size_min,
            auto_reranker_batch_size_max=auto_reranker_batch_size_max,
            auto_reranker_batch_size_sample_size=auto_reranker_batch_size_sample_size,
            auto_reranker_batch_size_memory_utilization=auto_reranker_batch_size_memory_utilization,
        )
    if _is_parquet_path(output_path):
        raise ValueError("Parquet output requires a parquet input_path")
    if only_verified:
        raise ValueError("--only-verified is only supported for parquet inputs")

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
    parser = argparse.ArgumentParser(
        description="Add stronger-reranker score fields to a FlagEmbedding-style JSONL or question/answer parquet."
    )
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
    parser.add_argument(
        "--parquet_score_column",
        type=str,
        default=_config_optional("STRONGER_RERANKER_PARQUET_SCORE_COLUMN", default=DEFAULT_PARQUET_SCORE_FIELD),
        help="Output score column for question/answer parquet inputs.",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default=_config_optional("STRONGER_RERANKER_PARQUET_QUESTION_COLUMN", default="question"),
        help="Question column for parquet inputs.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default=_config_optional("STRONGER_RERANKER_PARQUET_ANSWER_COLUMN", default="answer"),
        help="Answer column for parquet inputs.",
    )
    parser.add_argument(
        "--rejected_column",
        type=str,
        default=_config_optional("STRONGER_RERANKER_PARQUET_REJECTED_COLUMN", default="rejected"),
        help="Rejected/verification column used by --only-verified for parquet inputs.",
    )
    parser.add_argument("--score_negatives", action="store_true")
    parser.add_argument(
        "--only-verified",
        dest="only_verified",
        action="store_true",
        help="For parquet inputs, score only rows where rejected is False.",
    )
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
        parquet_score_column=args.parquet_score_column,
        question_column=args.question_column,
        answer_column=args.answer_column,
        rejected_column=args.rejected_column,
        only_verified=args.only_verified,
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
