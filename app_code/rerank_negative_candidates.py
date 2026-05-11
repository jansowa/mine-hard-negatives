import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from decouple import config
from tqdm import tqdm

from models import get_reranker_model, rerank

ADAPTIVE_RERANK_FIELDS = (
    ("final_percentile", pa.float32()),
    ("final_threshold_rank", pa.float32()),
    ("final_selected", pa.bool_()),
    ("final_selection_tier", pa.string()),
    ("final_rerank_budget", pa.int32()),
)


class AdaptiveQueryReportRow(TypedDict):
    query_id: str
    strict_final_negatives: int
    target_negatives: int
    candidate_rows: int
    scored_rows: int
    budget_limit: int


def _pack_pair(query_id: str, document_id: str) -> str:
    return f"{str(query_id)}\t{str(document_id)}"


def _json_safe(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _worker_file(output_path: str) -> str:
    output_dir = os.path.dirname(output_path) or "."
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(output_dir, f"{output_basename}_worker_0_0.jsonl")


def _load_scored_pairs_from_parquet(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()

    scored: set[str] = set()
    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return scored

    for batch in pf.iter_batches(columns=["query_id", "document_id"], batch_size=200_000):
        table = pa.Table.from_batches([batch])
        for query_id, document_id in zip(table.column("query_id").to_pylist(), table.column("document_id").to_pylist()):
            scored.add(_pack_pair(query_id, document_id))
    return scored


def _load_scored_pairs_from_jsonl(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()

    scored: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scored.add(_pack_pair(row["query_id"], row["document_id"]))
    return scored


def _load_scored_pairs(paths: Iterable[str]) -> set[str]:
    scored: set[str] = set()
    for path in paths:
        if path.endswith(".jsonl"):
            scored.update(_load_scored_pairs_from_jsonl(path))
        else:
            scored.update(_load_scored_pairs_from_parquet(path))
    return scored


def _stage_float_config(name: str, fallback_name: str, default: float) -> float:
    return config(name, cast=float, default=config(fallback_name, cast=float, default=default))


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _build_ecdf(sorted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(sorted_values) == 0:
        raise ValueError("Cannot build ECDF from an empty score array")
    ecdf_y = np.linspace(0.0, 1.0, len(sorted_values)) if len(sorted_values) > 1 else np.array([1.0])
    return sorted_values, ecdf_y


def _percentile_from_ecdf(values: np.ndarray | float, ecdf_x: np.ndarray, ecdf_y: np.ndarray):
    return np.interp(values, ecdf_x, ecdf_y, left=0.0, right=1.0)


def _inv_percentile_from_ecdf(percentiles: np.ndarray | float, ecdf_x: np.ndarray, ecdf_y: np.ndarray):
    percentiles = np.clip(percentiles, 0.0, 1.0)
    idx = np.searchsorted(ecdf_y, percentiles, side="right") - 1
    idx = np.clip(idx, 0, len(ecdf_x) - 1)
    return ecdf_x[idx]


def _load_final_thresholds(
    relevant_path: str,
    positive_score_column: str,
    beta: float,
    u_floor: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    relevant_df = pd.read_parquet(relevant_path, columns=["query_id", "document_id", positive_score_column])
    relevant_df["query_id"] = relevant_df["query_id"].astype("string")
    relevant_df[positive_score_column] = pd.to_numeric(relevant_df[positive_score_column], errors="coerce")

    positive_scores = np.sort(relevant_df[positive_score_column].dropna().to_numpy(copy=True))
    ecdf_x, ecdf_y = _build_ecdf(positive_scores)

    min_scores = relevant_df.groupby("query_id", dropna=False)[positive_score_column].min().dropna()
    u_pos = _percentile_from_ecdf(min_scores.to_numpy(copy=True), ecdf_x, ecdf_y)
    threshold_ranks = np.maximum(
        _inv_percentile_from_ecdf(beta * u_pos, ecdf_x, ecdf_y),
        _inv_percentile_from_ecdf(u_floor, ecdf_x, ecdf_y),
    )
    thresholds = {str(query_id): float(threshold) for query_id, threshold in zip(min_scores.index, threshold_ranks)}
    return thresholds, ecdf_x, ecdf_y


def _load_scored_rows_from_jsonl(path: str, ranking_column: str) -> dict[str, dict]:
    if not os.path.isfile(path):
        return {}

    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if ranking_column not in row and "ranking" not in row:
                continue
            rows[_pack_pair(row["query_id"], row["document_id"])] = row
    return rows


def _load_scored_rows_from_parquet(path: str, ranking_column: str) -> dict[str, dict]:
    if not os.path.isfile(path):
        return {}

    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return {}

    rows: dict[str, dict] = {}
    columns = None
    schema_names = set(pf.schema_arrow.names)
    if "query_id" in schema_names and "document_id" in schema_names:
        columns = list(schema_names)
    for batch in pf.iter_batches(columns=columns, batch_size=200_000):
        table = pa.Table.from_batches([batch])
        for row in table.to_pylist():
            if ranking_column not in row and "ranking" not in row:
                continue
            rows[_pack_pair(row["query_id"], row["document_id"])] = row
    return rows


def _load_scored_rows(paths: Iterable[str], ranking_column: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        if path.endswith(".jsonl"):
            rows.update(_load_scored_rows_from_jsonl(path, ranking_column))
        else:
            rows.update(_load_scored_rows_from_parquet(path, ranking_column))
    return rows


def _output_schema(candidates_path: str, ranking_column: str, include_adaptive_fields: bool = False) -> pa.Schema:
    candidate_schema = pq.ParquetFile(candidates_path).schema_arrow
    fields = []
    final_columns = {ranking_column, "ranking"}
    if include_adaptive_fields:
        final_columns.update(name for name, _ in ADAPTIVE_RERANK_FIELDS)
    for field in candidate_schema:
        if field.name not in final_columns:
            fields.append(field)

    if ranking_column != "ranking":
        fields.append(pa.field(ranking_column, pa.float32()))
    fields.append(pa.field("ranking", pa.float32()))
    if include_adaptive_fields:
        existing_names = {field.name for field in fields}
        fields.extend(
            pa.field(name, field_type) for name, field_type in ADAPTIVE_RERANK_FIELDS if name not in existing_names
        )
    return pa.schema(fields)


def _row_for_output(
    candidate_row: dict,
    score: float,
    ranking_column: str,
    schema: pa.Schema,
    extra_values: dict | None = None,
) -> dict:
    out = {name: _json_safe(candidate_row.get(name)) for name in schema.names}
    if ranking_column != "ranking":
        out[ranking_column] = float(score)
    out["ranking"] = float(score)
    if extra_values:
        for key, value in extra_values.items():
            if key in out:
                out[key] = _json_safe(value)
    return out


def _write_jsonl(handle, rows: list[dict]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def consolidate_worker_jsonl(worker_file: str, output_path: str, schema: pa.Schema, row_group_size: int) -> None:
    if not os.path.isfile(worker_file):
        print(f"No worker file found at {worker_file}; nothing to consolidate.")
        return

    temp_output_path = f"{output_path}.tmp"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    writer = None
    total_rows = 0

    def write_buffer(buffer: list[dict]) -> None:
        nonlocal writer, total_rows
        if not buffer:
            return
        normalised = [{name: row.get(name) for name in schema.names} for row in buffer]
        table = pa.Table.from_pylist(normalised, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(temp_output_path, schema=schema, compression="zstd", use_dictionary=True)
        writer.write_table(table)
        total_rows += len(buffer)

    try:
        buffer: list[dict] = []
        with open(worker_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                buffer.append(json.loads(line))
                if len(buffer) >= row_group_size:
                    write_buffer(buffer)
                    buffer = []
        write_buffer(buffer)

        if writer is not None:
            writer.close()
            os.replace(temp_output_path, output_path)
            print(f"Saved {total_rows:,} final reranked rows to {output_path}")
        else:
            print("No rows found to consolidate.")
    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise


def _strict_final_count(rows: Iterable[dict], ranking_column: str, threshold: float) -> int:
    doc_ids = set()
    for row in rows:
        score = row.get(ranking_column, row.get("ranking"))
        if score is None:
            continue
        if float(score) <= threshold:
            doc_ids.add(str(row["document_id"]))
    return len(doc_ids)


def _candidate_score_column(candidates_df: pd.DataFrame, preferred_column: str | None) -> str:
    candidates = [preferred_column, "candidate_ranking", "ranking", "retrieval_score"]
    for column in candidates:
        if column and column in candidates_df.columns:
            return column
    raise ValueError("Could not find a candidate score column in candidates_path")


def _prepare_candidates(candidates_df: pd.DataFrame, candidate_score_column: str, candidate_selected_column: str):
    candidates_df["query_id"] = candidates_df["query_id"].astype("string")
    candidates_df["document_id"] = candidates_df["document_id"].astype("string")
    candidates_df["_candidate_score"] = pd.to_numeric(candidates_df[candidate_score_column], errors="coerce")
    if candidate_selected_column in candidates_df.columns:
        candidates_df["_candidate_selected"] = candidates_df[candidate_selected_column].fillna(False).astype(bool)
    else:
        candidates_df["_candidate_selected"] = True
    selected = candidates_df[candidates_df["_candidate_selected"]].sort_values(
        ["query_id", "_candidate_score"], ascending=[True, False], na_position="last"
    )
    unselected = candidates_df[~candidates_df["_candidate_selected"]].sort_values(
        ["query_id", "_candidate_score"], ascending=[True, True], na_position="last"
    )
    candidates_df = pd.concat([selected, unselected], ignore_index=True)
    return candidates_df.drop_duplicates(subset=["query_id", "document_id"], keep="first")


def _write_report(report_path: str | None, report: dict) -> None:
    if not report_path:
        return
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def rerank_candidates_adaptive(
    candidates_path: str,
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    output_path: str,
    reranker_model_name: str,
    reranker_batch_size: int,
    ranking_column: str,
    candidate_selected_column: str,
    candidate_score_column: str | None,
    positive_score_column: str,
    num_negatives: int,
    beta: float,
    u_floor: float,
    initial_budget: int,
    budget_step: int,
    max_budget: int,
    resume: bool,
    row_group_size: int,
    report_path: str | None,
) -> None:
    num_negatives = _positive_int(num_negatives, "num_negatives")
    initial_budget = _positive_int(initial_budget, "initial_budget")
    budget_step = _positive_int(budget_step, "budget_step")
    max_budget = _positive_int(max_budget, "max_budget")
    if initial_budget > max_budget:
        raise ValueError("initial_budget must be less than or equal to max_budget")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    worker_file = _worker_file(output_path)
    if not resume and os.path.exists(worker_file):
        os.remove(worker_file)

    print(f"Loading final thresholds from {relevant_path}")
    thresholds, ecdf_x, ecdf_y = _load_final_thresholds(
        relevant_path=relevant_path,
        positive_score_column=positive_score_column,
        beta=beta,
        u_floor=u_floor,
    )

    print(f"Loading queries from {queries_path}")
    queries_df = pd.read_parquet(queries_path, columns=["id", "text"])
    queries_df["id"] = queries_df["id"].astype("string")
    queries = dict(zip(queries_df["id"].astype(str), queries_df["text"].astype(str)))

    print(f"Loading corpus from {corpus_path}")
    corpus_df = pd.read_parquet(corpus_path, columns=["id", "text"])
    corpus_df["id"] = corpus_df["id"].astype("string")
    corpus = dict(zip(corpus_df["id"].astype(str), corpus_df["text"].astype(str)))

    print(f"Loading candidate metadata from {candidates_path}")
    candidates_df = pd.read_parquet(candidates_path)
    score_column = _candidate_score_column(candidates_df, candidate_score_column)
    candidates_df = _prepare_candidates(candidates_df, score_column, candidate_selected_column)
    candidates_df = candidates_df[candidates_df["query_id"].astype(str).isin(thresholds)]

    already_scored_rows = _load_scored_rows([worker_file, output_path], ranking_column) if resume else {}
    if already_scored_rows:
        print(f"Resume enabled: found {len(already_scored_rows):,} already reranked pairs")

    scored_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in already_scored_rows.values():
        scored_by_query[str(row["query_id"])].append(row)

    tokenizer, reranker_model = get_reranker_model(reranker_model_name)
    print("Final reranker loaded.")

    output_schema = _output_schema(candidates_path, ranking_column, include_adaptive_fields=True)
    newly_scored = 0
    skipped_missing_text = 0
    query_report: list[AdaptiveQueryReportRow] = []
    strict_histogram: Counter[int] = Counter()

    grouped = candidates_df.groupby("query_id", sort=False)
    with open(worker_file, "a", encoding="utf-8") as output_handle:
        with tqdm(total=len(grouped), unit="query", desc="Adaptive final reranking") as pbar:
            for query_id, group in grouped:
                qid = str(query_id)
                threshold = thresholds.get(qid)
                if threshold is None:
                    pbar.update(1)
                    continue

                existing_rows = scored_by_query.get(qid, [])
                existing_doc_ids = {str(row["document_id"]) for row in existing_rows}
                strict_count = _strict_final_count(existing_rows, ranking_column, threshold)
                budget_limit = max(initial_budget, min(max_budget, len(existing_doc_ids)))
                candidate_rows = group.to_dict("records")

                while strict_count < num_negatives and budget_limit <= max_budget:
                    window = candidate_rows[: min(budget_limit, len(candidate_rows))]
                    rows_to_score = [row for row in window if str(row["document_id"]) not in existing_doc_ids]
                    if not rows_to_score:
                        if budget_limit >= max_budget or len(window) >= len(candidate_rows):
                            break
                        budget_limit = min(max_budget, budget_limit + budget_step)
                        continue

                    query_text = queries.get(qid)
                    rerank_queries: list[str] = []
                    rerank_docs: list[str] = []
                    valid_rows: list[dict[str, Any]] = []
                    for candidate_row in rows_to_score:
                        document_id = str(candidate_row["document_id"])
                        document_text = corpus.get(document_id)
                        if query_text is None or document_text is None:
                            skipped_missing_text += 1
                            existing_doc_ids.add(document_id)
                            continue
                        rerank_queries.append(query_text)
                        rerank_docs.append(document_text)
                        valid_rows.append(candidate_row)

                    if not valid_rows:
                        if budget_limit >= max_budget or len(window) >= len(candidate_rows):
                            break
                        budget_limit = min(max_budget, budget_limit + budget_step)
                        continue

                    scores = rerank(
                        tokenizer,
                        reranker_model,
                        rerank_queries,
                        rerank_docs,
                        batch_size=reranker_batch_size,
                        model_name=reranker_model_name,
                    )
                    percentiles = _percentile_from_ecdf(np.array(scores, dtype=float), ecdf_x, ecdf_y)

                    rows = []
                    for candidate_row, score, percentile in zip(valid_rows, scores, percentiles):
                        final_selected = float(score) <= threshold
                        extra_values = {
                            "final_percentile": float(percentile),
                            "final_threshold_rank": float(threshold),
                            "final_selected": bool(final_selected),
                            "final_selection_tier": "strict" if final_selected else "candidate",
                            "final_rerank_budget": int(budget_limit),
                        }
                        row = _row_for_output(candidate_row, score, ranking_column, output_schema, extra_values)
                        rows.append(row)
                        existing_doc_ids.add(str(row["document_id"]))
                        scored_by_query[qid].append(row)
                        if final_selected:
                            strict_count += 1

                    _write_jsonl(output_handle, rows)
                    newly_scored += len(rows)

                    if strict_count < num_negatives:
                        if budget_limit >= max_budget or budget_limit >= len(candidate_rows):
                            break
                        budget_limit = min(max_budget, budget_limit + budget_step)

                strict_histogram[min(strict_count, num_negatives)] += 1
                query_report.append(
                    {
                        "query_id": qid,
                        "strict_final_negatives": int(strict_count),
                        "target_negatives": int(num_negatives),
                        "candidate_rows": int(len(candidate_rows)),
                        "scored_rows": int(len(existing_doc_ids)),
                        "budget_limit": int(min(budget_limit, max_budget)),
                    }
                )
                pbar.update(1)

    print(f"Newly reranked pairs: {newly_scored:,}")
    consolidate_worker_jsonl(worker_file, output_path, output_schema, row_group_size=row_group_size)

    complete_queries = sum(1 for row in query_report if row["strict_final_negatives"] >= num_negatives)
    report = {
        "mode": "adaptive",
        "target_negatives": num_negatives,
        "beta": beta,
        "u_floor": u_floor,
        "initial_budget": initial_budget,
        "budget_step": budget_step,
        "max_budget": max_budget,
        "queries_with_candidates": len(query_report),
        "complete_queries": complete_queries,
        "partial_queries": len(query_report) - complete_queries,
        "newly_scored_pairs": newly_scored,
        "skipped_missing_text": skipped_missing_text,
        "strict_final_negatives_histogram": {str(key): value for key, value in sorted(strict_histogram.items())},
        "worst_partial_queries": sorted(
            (row for row in query_report if row["strict_final_negatives"] < num_negatives),
            key=lambda row: (row["strict_final_negatives"], -row["candidate_rows"]),
        )[:50],
    }
    _write_report(report_path, report)


def rerank_candidates(
    candidates_path: str,
    queries_path: str,
    corpus_path: str,
    output_path: str,
    reranker_model_name: str,
    reranker_batch_size: int,
    ranking_column: str = "final_ranking",
    candidate_selected_column: str = "candidate_selected",
    selected_only: bool = True,
    chunk_size: int = 100_000,
    resume: bool = True,
    row_group_size: int = 100_000,
    rerank_mode: str = "selected",
    relevant_path: str | None = None,
    candidate_score_column: str | None = None,
    positive_score_column: str = "positive_ranking",
    num_negatives: int = 10,
    beta: float = 0.01,
    u_floor: float = 0.005,
    initial_budget: int = 20,
    budget_step: int = 10,
    max_budget: int = 80,
    report_path: str | None = None,
) -> None:
    if not ranking_column:
        raise ValueError("ranking_column must not be empty")
    if reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size must be greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be greater than 0")

    if rerank_mode not in {"selected", "all", "adaptive"}:
        raise ValueError("rerank_mode must be one of: selected, all, adaptive")
    if rerank_mode == "adaptive":
        if relevant_path is None:
            raise ValueError("relevant_path is required when rerank_mode='adaptive'")
        rerank_candidates_adaptive(
            candidates_path=candidates_path,
            queries_path=queries_path,
            corpus_path=corpus_path,
            relevant_path=relevant_path,
            output_path=output_path,
            reranker_model_name=reranker_model_name,
            reranker_batch_size=reranker_batch_size,
            ranking_column=ranking_column,
            candidate_selected_column=candidate_selected_column,
            candidate_score_column=candidate_score_column,
            positive_score_column=positive_score_column,
            num_negatives=num_negatives,
            beta=beta,
            u_floor=u_floor,
            initial_budget=initial_budget,
            budget_step=budget_step,
            max_budget=max_budget,
            resume=resume,
            row_group_size=row_group_size,
            report_path=report_path,
        )
        return

    if rerank_mode == "all":
        selected_only = False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    worker_file = _worker_file(output_path)
    if not resume and os.path.exists(worker_file):
        os.remove(worker_file)

    print(f"Loading queries from {queries_path}")
    queries_df = pd.read_parquet(queries_path, columns=["id", "text"])
    queries_df["id"] = queries_df["id"].astype("string")
    queries_df = queries_df.rename(columns={"text": "query_text"}).set_index("id")

    print(f"Loading corpus from {corpus_path}")
    corpus_df = pd.read_parquet(corpus_path, columns=["id", "text"])
    corpus_df["id"] = corpus_df["id"].astype("string")
    corpus_df = corpus_df.rename(columns={"text": "document_text"}).set_index("id")

    already_scored = _load_scored_pairs([worker_file, output_path]) if resume else set()
    if already_scored:
        print(f"Resume enabled: found {len(already_scored):,} already reranked pairs")

    tokenizer, reranker_model = get_reranker_model(reranker_model_name)
    print("Final reranker loaded.")

    output_schema = _output_schema(candidates_path, ranking_column)
    pf = pq.ParquetFile(candidates_path)
    total_rows = pf.metadata.num_rows
    newly_scored = 0

    with open(worker_file, "a", encoding="utf-8") as output_handle:
        with tqdm(total=total_rows, unit="row", desc="Final reranking candidates") as pbar:
            for batch in pf.iter_batches(batch_size=chunk_size):
                candidates_df = batch.to_pandas()
                pbar.update(len(candidates_df))
                if candidates_df.empty:
                    continue

                candidates_df["query_id"] = candidates_df["query_id"].astype("string")
                candidates_df["document_id"] = candidates_df["document_id"].astype("string")

                if selected_only and candidate_selected_column in candidates_df.columns:
                    candidates_df = candidates_df[candidates_df[candidate_selected_column].fillna(False)]
                if candidates_df.empty:
                    continue

                if already_scored:
                    packed = (
                        _pack_pair(query_id, document_id)
                        for query_id, document_id in zip(
                            candidates_df["query_id"].values, candidates_df["document_id"].values
                        )
                    )
                    candidates_df = candidates_df.loc[[key not in already_scored for key in packed]]
                if candidates_df.empty:
                    continue

                merged_df = pd.merge(
                    candidates_df,
                    queries_df,
                    left_on="query_id",
                    right_index=True,
                    how="inner",
                )
                merged_df = pd.merge(
                    merged_df,
                    corpus_df,
                    left_on="document_id",
                    right_index=True,
                    how="inner",
                )
                if merged_df.empty:
                    continue

                scores = rerank(
                    tokenizer,
                    reranker_model,
                    merged_df["query_text"].values.tolist(),
                    merged_df["document_text"].values.tolist(),
                    batch_size=reranker_batch_size,
                    model_name=reranker_model_name,
                )

                rows = [
                    _row_for_output(candidate_row, score, ranking_column, output_schema)
                    for candidate_row, score in zip(merged_df.to_dict("records"), scores)
                ]
                _write_jsonl(output_handle, rows)
                newly_scored += len(rows)
                already_scored.update(_pack_pair(row["query_id"], row["document_id"]) for row in rows)

    print(f"Newly reranked pairs: {newly_scored:,}")
    consolidate_worker_jsonl(worker_file, output_path, output_schema, row_group_size=row_group_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-rerank lightweight negative candidates.")
    num_negatives_default = config("NUM_NEGATIVES", cast=int, default=10)
    parser.add_argument(
        "--candidates_path",
        type=str,
        default=config("NEGATIVE_CANDIDATES_PATH", default="data/negative_candidates.parquet"),
    )
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--output_path", type=str, default=config("NEGATIVES_PATH"))
    parser.add_argument(
        "--reranker_model_name", type=str, default=config("FINAL_RERANKER_NAME", default=config("RERANKER_NAME"))
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=config("FINAL_RERANKER_BATCH_SIZE", cast=int, default=config("RERANKER_BATCH_SIZE", cast=int)),
    )
    parser.add_argument(
        "--ranking_column",
        type=str,
        default=config("FINAL_RANKING_COLUMN", default="final_ranking"),
        help="Final score column name. The output also writes ranking as a compatibility alias.",
    )
    parser.add_argument(
        "--candidate_selected_column",
        type=str,
        default=config("CANDIDATE_SELECTED_COLUMN", default="candidate_selected"),
    )
    parser.add_argument(
        "--candidate_score_column",
        type=str,
        default=config("NEGATIVE_RANKING_COLUMN", default="candidate_ranking"),
        help="Small-reranker score column used to prioritize adaptive final reranking.",
    )
    parser.add_argument(
        "--rerank_mode",
        choices=["adaptive", "selected", "all"],
        default=config("FINAL_RERANK_MODE", default="adaptive"),
        help="adaptive reranks more candidates only for queries that still need final negatives.",
    )
    parser.add_argument(
        "--relevant_path",
        type=str,
        default=config("FINAL_POSITIVE_RANKS_OUTPUT_PATH", default=config("RELEVANT_WITH_SCORE_PATH")),
        help="Relevant-with-final-positive-scores file used by adaptive mode.",
    )
    parser.add_argument(
        "--positive_score_column",
        type=str,
        default=config(
            "FINAL_POSITIVE_SCORE_COLUMN", default=config("POSITIVE_SCORE_COLUMN", default="positive_ranking")
        ),
        help="Final positive score column used by adaptive mode.",
    )
    parser.add_argument("--num_negatives", type=int, default=num_negatives_default)
    parser.add_argument("--beta", type=float, default=_stage_float_config("FINAL_BETA", "BETA", 0.01))
    parser.add_argument("--u_floor", type=float, default=_stage_float_config("FINAL_U_FLOOR", "U_FLOOR", 0.005))
    parser.add_argument(
        "--initial_budget",
        type=int,
        default=config("FINAL_RERANK_INITIAL_BUDGET", cast=int, default=max(1, num_negatives_default * 2)),
    )
    parser.add_argument(
        "--budget_step",
        type=int,
        default=config("FINAL_RERANK_BUDGET_STEP", cast=int, default=max(1, num_negatives_default)),
    )
    parser.add_argument(
        "--max_budget",
        type=int,
        default=config("FINAL_RERANK_MAX_BUDGET", cast=int, default=max(1, num_negatives_default * 8)),
    )
    parser.add_argument("--report_path", type=str, default=config("FINAL_RERANK_REPORT_PATH", default=None))
    parser.add_argument("--selected-only", dest="selected_only", action="store_true")
    parser.add_argument("--all-candidates", dest="selected_only", action="store_false")
    parser.set_defaults(selected_only=config("FINAL_RERANK_SELECTED_ONLY", cast=bool, default=True))
    parser.add_argument("--chunk_size", type=int, default=config("PROCESSING_CHUNK_SIZE", cast=int, default=100_000))
    parser.add_argument(
        "--row_group_size",
        type=int,
        default=config("NEGATIVES_PARQUET_ROW_GROUP_SIZE", cast=int, default=100_000),
    )
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=config("FINAL_RERANK_RESUME", cast=bool, default=True))
    args = parser.parse_args()
    report_path = args.report_path
    if args.rerank_mode == "adaptive" and report_path is None:
        report_path = f"{args.output_path}.report.json"

    rerank_candidates(
        candidates_path=args.candidates_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        reranker_model_name=args.reranker_model_name,
        reranker_batch_size=args.reranker_batch_size,
        ranking_column=args.ranking_column,
        candidate_selected_column=args.candidate_selected_column,
        selected_only=args.selected_only,
        chunk_size=args.chunk_size,
        resume=args.resume,
        row_group_size=args.row_group_size,
        rerank_mode=args.rerank_mode,
        relevant_path=args.relevant_path,
        candidate_score_column=args.candidate_score_column,
        positive_score_column=args.positive_score_column,
        num_negatives=args.num_negatives,
        beta=args.beta,
        u_floor=args.u_floor,
        initial_budget=args.initial_budget,
        budget_step=args.budget_step,
        max_budget=args.max_budget,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
