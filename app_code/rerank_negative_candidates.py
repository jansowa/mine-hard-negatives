import argparse
import json
import os
from collections.abc import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from decouple import config
from tqdm import tqdm

from models import get_reranker_model, rerank


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


def _output_schema(candidates_path: str, ranking_column: str) -> pa.Schema:
    candidate_schema = pq.ParquetFile(candidates_path).schema_arrow
    fields = []
    final_columns = {ranking_column, "ranking"}
    for field in candidate_schema:
        if field.name not in final_columns:
            fields.append(field)

    if ranking_column != "ranking":
        fields.append(pa.field(ranking_column, pa.float32()))
    fields.append(pa.field("ranking", pa.float32()))
    return pa.schema(fields)


def _row_for_output(candidate_row: dict, score: float, ranking_column: str, schema: pa.Schema) -> dict:
    out = {name: _json_safe(candidate_row.get(name)) for name in schema.names}
    if ranking_column != "ranking":
        out[ranking_column] = float(score)
    out["ranking"] = float(score)
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
) -> None:
    if not ranking_column:
        raise ValueError("ranking_column must not be empty")
    if reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size must be greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be greater than 0")

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
    )


if __name__ == "__main__":
    main()
