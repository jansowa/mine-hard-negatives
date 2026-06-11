import os

os.environ.setdefault("POLARS_MAX_THREADS", "12")

import argparse
import json
import re
import sqlite3
from collections import Counter

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from decouple import config
from tqdm.auto import tqdm

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# --------------------------- SQLite utils ---------------------------


def sqlite_connect(db_path: str) -> sqlite3.Connection:
    d = os.path.dirname(db_path)
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("PRAGMA mmap_size=3000000000;")
    return conn


def ensure_corpus_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus(
            id   TEXT PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)
    conn.commit()


def ensure_negcount_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS neg_count(
            doc_id TEXT PRIMARY KEY,
            c      INTEGER NOT NULL
        );
    """)
    conn.commit()


def bulk_upsert_corpus(conn: sqlite3.Connection, rows, batch_size: int = 50_000):
    cur = conn.cursor()
    cur.execute("BEGIN")
    i = 0
    for rid, text in rows:
        cur.execute("INSERT OR REPLACE INTO corpus(id, text) VALUES (?, ?)", (rid, text))
        i += 1
        if i % batch_size == 0:
            conn.commit()
            cur.execute("BEGIN")
    conn.commit()


def fetch_texts_batch(conn: sqlite3.Connection, ids: list[str], batch: int = 2000) -> dict[str, str]:
    res: dict[str, str] = {}
    if not ids:
        return res
    cur = conn.cursor()
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        q = "SELECT id, text FROM corpus WHERE id IN ({})".format(",".join(["?"] * len(chunk)))
        cur.execute(q, chunk)
        rows = cur.fetchall()
        if rows:
            res.update({rid: txt for rid, txt in rows})
    return res


def fetch_counts_batch(conn: sqlite3.Connection, ids: list[str], batch: int = 5000) -> dict[str, int]:
    res: dict[str, int] = {}
    if not ids:
        return res
    cur = conn.cursor()
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        q = "SELECT doc_id, c FROM neg_count WHERE doc_id IN ({})".format(",".join(["?"] * len(chunk)))
        cur.execute(q, chunk)
        rows = cur.fetchall()
        if rows:
            res.update({rid: c for rid, c in rows})
    return res


def inc_counts_batch(conn: sqlite3.Connection, ids: list[str]):
    if not ids:
        return
    cnt = Counter(ids)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO neg_count(doc_id, c) VALUES(?, ?) ON CONFLICT(doc_id) DO UPDATE SET c = c + excluded.c",
        list(cnt.items()),
    )


def reset_neg_counts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM neg_count")
    conn.commit()


def write_report(report_path: str | None, report: dict) -> None:
    if not report_path:
        return
    d = os.path.dirname(report_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------- ECDF helpers ---------------------------


def build_ecdf(sorted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("Empty array for ECDF.")
    ecdf_x = sorted_values
    ecdf_y = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])
    return ecdf_x, ecdf_y


def percentile_from_ecdf(v: np.ndarray | float, ecdf_x: np.ndarray, ecdf_y: np.ndarray):
    return np.interp(v, ecdf_x, ecdf_y, left=0.0, right=1.0)


def inv_percentile_from_ecdf(p: np.ndarray | float, ecdf_x: np.ndarray, ecdf_y: np.ndarray):
    p = np.clip(p, 0.0, 1.0)
    # wektorowo: dla każdego p zwróć ecdf_x[max idx, gdzie ecdf_y <= p]
    idx = np.searchsorted(ecdf_y, p, side="right") - 1
    idx = np.clip(idx, 0, len(ecdf_x) - 1)
    return ecdf_x[idx]


def token_counts(text: str) -> Counter[str]:
    return Counter(WORD_RE.findall((text or "").casefold()))


def token_overlap_ratio(left: str, right: str) -> float:
    left_counts = token_counts(left)
    right_counts = token_counts(right)
    denominator = min(sum(left_counts.values()), sum(right_counts.values()))
    if denominator == 0:
        return 1.0 if left_counts == right_counts else 0.0
    overlap = sum((left_counts & right_counts).values())
    return overlap / denominator


def is_near_duplicate(candidate_text: str, reference_texts: list[str], threshold: float) -> bool:
    if not candidate_text.strip():
        return True
    return any(token_overlap_ratio(candidate_text, reference_text) >= threshold for reference_text in reference_texts)


# --------------------------- Build corpus SQLite ---------------------------


def build_or_load_corpus_sqlite(
    corpus_path: str,
    corpus_sqlite_path: str,
    id_col="id",
    text_col="text",
    block_rows: int = 1_000_000,
    low_memory_optimizations: bool = False,
):
    conn = sqlite_connect(corpus_sqlite_path)
    ensure_corpus_db(conn)

    parquet_file = pq.ParquetFile(corpus_path)
    expected_rows = parquet_file.metadata.num_rows
    cur = conn.execute("SELECT COUNT(1) FROM corpus;")
    existing_rows = int(cur.fetchone()[0])
    if existing_rows == expected_rows:
        conn.close()
        return

    if low_memory_optimizations:
        with tqdm(total=expected_rows, desc="Budowanie bazy korpusu (SQLite)", unit="rows") as pbar:
            for batch in parquet_file.iter_batches(columns=[id_col, text_col], batch_size=block_rows):
                ids = batch.column(batch.schema.get_field_index(id_col)).to_pylist()
                texts = batch.column(batch.schema.get_field_index(text_col)).to_pylist()
                rows = (
                    (str(row_id), "" if text is None else str(text))
                    for row_id, text in zip(ids, texts)
                )
                bulk_upsert_corpus(conn, rows)
                pbar.update(batch.num_rows)
    else:
        dataset = pl.scan_parquet(corpus_path).select(
            [pl.col(id_col).cast(pl.Utf8), pl.col(text_col).cast(pl.Utf8)]
        )
        frame = dataset.collect()
        with tqdm(total=frame.height, desc="Budowanie bazy korpusu (SQLite)", unit="rows") as pbar:
            for start in range(0, frame.height, block_rows):
                end = min(start + block_rows, frame.height)
                chunk = frame.slice(start, end - start)
                bulk_upsert_corpus(conn, zip(chunk[id_col].to_list(), chunk[text_col].to_list()))
                pbar.update(end - start)

    conn.close()


# --------------------------- Główny pipeline ---------------------------


def process_negatives_streaming(
    corpus_path: str,
    queries_path: str,
    relevant_path: str,
    negatives_path: str,
    output_path: str,
    num_negatives: int,
    beta: float,
    u_floor: float,
    max_neg_reuse: int,
    corpus_sqlite_path: str,
    negcount_sqlite_path: str,
    query_chunk_size: int,
    oversample_factor: int,
    positive_score_column: str = "positive_ranking",
    negative_score_column: str = "ranking",
    positive_original_score_column: str | None = None,
    negative_original_score_column: str | None = None,
    prompt: str = "",
    dataset_type: str = "retrieval",
    backfill_policy: str = "relaxed",
    report_path: str | None = None,
    mine_positives: bool = False,
    max_mined_positives: int = 1,
    u_sanity_ceiling: float = 0.90,
    u_absolute_ceiling: float = 0.995,
    u_positive_beta: float = 0.95,
    positive_near_duplicate_threshold: float = 0.80,
    query_skip: int = 0,
    query_limit: int | None = None,
    low_memory_optimizations: bool = False,
):
    if not positive_score_column:
        raise ValueError("positive_score_column must not be empty")
    if not negative_score_column:
        raise ValueError("negative_score_column must not be empty")
    if backfill_policy not in {"none", "relaxed"}:
        raise ValueError("backfill_policy must be one of: none, relaxed")
    if max_mined_positives < 0:
        raise ValueError("max_mined_positives must be greater than or equal to 0")
    if query_skip < 0:
        raise ValueError("query_skip must be greater than or equal to 0")
    if query_limit is not None and query_limit < 0:
        raise ValueError("query_limit must be greater than or equal to 0 or None")
    if query_limit == 0:
        query_limit = None
    for name, value in (
        ("u_sanity_ceiling", u_sanity_ceiling),
        ("u_absolute_ceiling", u_absolute_ceiling),
        ("positive_near_duplicate_threshold", positive_near_duplicate_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if u_positive_beta < 0.0:
        raise ValueError("u_positive_beta must be greater than or equal to 0")

    try:
        pl.enable_string_cache()
    except AttributeError:
        pass

    build_or_load_corpus_sqlite(
        corpus_path=corpus_path,
        corpus_sqlite_path=corpus_sqlite_path,
        block_rows=1_000_000,
        low_memory_optimizations=low_memory_optimizations,
    )

    neg_conn = sqlite_connect(negcount_sqlite_path)
    ensure_negcount_db(neg_conn)
    reset_neg_counts(neg_conn)

    qdf = pl.read_parquet(queries_path, columns=["id", "text"]).with_columns(
        pl.col("id").cast(pl.Utf8), pl.col("text").cast(pl.Utf8)
    )
    ordered_query_ids = qdf["id"].to_list()
    queries_dict = dict(zip(qdf["id"].to_list(), qdf["text"].to_list()))
    del qdf

    relevant_select = [
        pl.col("query_id"),
        pl.col("document_id"),
        pl.col(positive_score_column).alias("positive_ranking"),
    ]
    if positive_original_score_column:
        relevant_select.append(pl.col(positive_original_score_column).alias("original_positive_ranking"))
    relevant_casts = [
        pl.col("query_id").cast(pl.Utf8),
        pl.col("document_id").cast(pl.Utf8),
        pl.col("positive_ranking").cast(pl.Float64),
    ]
    if positive_original_score_column:
        relevant_casts.append(pl.col("original_positive_ranking").cast(pl.Float64))

    rel = pl.scan_parquet(relevant_path).select(relevant_select).with_columns(relevant_casts)

    pos_scores_all = np.sort(
        rel.select(pl.col("positive_ranking")).collect()["positive_ranking"].drop_nulls().to_numpy().copy()
    )
    if pos_scores_all.size == 0:
        raise ValueError("Brak wartości 'positive_ranking' w pliku RELEVANT_WITH_SCORE_PATH.")
    ecdf_x, ecdf_y = build_ecdf(pos_scores_all)

    try:
        target_scores = np.arange(5.0, 28.0 + 1e-9, 1.0, dtype=float)  # 5.0 .. 28.0
        target_p = percentile_from_ecdf(target_scores, ecdf_x, ecdf_y)
        print("Percentile map (score -> percentile in [0,1]):")
        for s, p in zip(target_scores.tolist(), target_p.tolist()):
            print(f"{s:>4.1f} -> {p:.6f}")
    except Exception as e:
        print(f"Nie udało się wypisać mapy percentyli: {e}")

    pos_aggs = [
        pl.col("document_id").implode().alias("pos_ids"),
        pl.col("positive_ranking").implode().alias("pos_scores"),
        pl.col("positive_ranking").min().alias("min_score"),
        pl.col("positive_ranking").max().alias("max_score"),
    ]
    if positive_original_score_column:
        pos_aggs.append(pl.col("original_positive_ranking").implode().alias("original_pos_scores"))

    pos_grouped = rel.group_by("query_id").agg(pos_aggs).with_columns(pl.col("min_score").cast(pl.Float64)).collect()

    min_scores = pos_grouped["min_score"].to_numpy()
    u_pos = percentile_from_ecdf(min_scores, ecdf_x, ecdf_y)
    u_strongest_pos = percentile_from_ecdf(pos_grouped["max_score"].to_numpy(), ecdf_x, ecdf_y)
    thr1 = inv_percentile_from_ecdf(beta * u_pos, ecdf_x, ecdf_y)
    thr2 = inv_percentile_from_ecdf(u_floor, ecdf_x, ecdf_y)
    thresholds = np.maximum(thr1, thr2)

    threshold_frame = {
        "query_id": pos_grouped["query_id"],
        "pos_ids": pos_grouped["pos_ids"],
        "pos_scores": pos_grouped["pos_scores"],
        "min_score": pos_grouped["min_score"],
        "u_strongest_pos": pl.Series(u_strongest_pos),
        "u_pos": pl.Series(u_pos),
        "threshold_rank": pl.Series(thresholds),
    }
    if positive_original_score_column:
        threshold_frame["original_pos_scores"] = pos_grouped["original_pos_scores"]
    thr_df = pl.DataFrame(threshold_frame)

    pos_map: dict[str, dict] = {
        qid: row for qid, row in zip(thr_df["query_id"], thr_df.iter_rows(named=True))
    }

    negative_select = [
        pl.col("query_id").cast(pl.Utf8),
        pl.col("document_id").cast(pl.Utf8),
        pl.col(negative_score_column).cast(pl.Float64).alias("ranking"),
    ]
    if negative_original_score_column:
        negative_select.append(pl.col(negative_original_score_column).cast(pl.Float64).alias("original_negative_ranking"))

    neg_scan = pl.scan_parquet(negatives_path).select(negative_select)

    eligible_qids = set(thr_df["query_id"].to_list())
    all_qids = [qid for qid in ordered_query_ids if qid in eligible_qids]
    end = None if query_limit is None else query_skip + query_limit
    all_qids = all_qids[query_skip:end]
    total_q = len(all_qids)

    out_f = open(output_path, "w", encoding="utf-8")
    corpus_db_path = os.path.abspath(corpus_sqlite_path)
    stats: Counter[str] = Counter()
    neg_count_histogram: Counter[int] = Counter()

    for start in tqdm(range(0, total_q, query_chunk_size), desc="Negatives streaming", unit="q_chunk"):
        end = min(start + query_chunk_size, total_q)
        qids_chunk = all_qids[start:end]

        thr_chunk = pl.DataFrame(
            {
                "query_id": qids_chunk,
                "threshold_rank": [pos_map[qid]["threshold_rank"] for qid in qids_chunk],
            }
        )

        pair_fields = ["document_id", "ranking", "strict_negative"]
        if negative_original_score_column:
            pair_fields.append("original_negative_ranking")

        candidates = (
            neg_scan.join(thr_chunk.lazy(), on="query_id", how="inner")
            .with_columns((pl.col("ranking") <= pl.col("threshold_rank")).alias("strict_negative"))
            .group_by("query_id")
            .agg(pl.struct(pair_fields).alias("pairs"))
            .collect()
        )

        if candidates.height == 0:
            stats["queries_without_candidates"] += len(qids_chunk)
            continue
        stats["queries_without_candidates"] += len(set(qids_chunk) - set(candidates["query_id"].to_list()))

        need_ids_set = set()
        for pairs in candidates["pairs"].to_list():
            for p in pairs:
                need_ids_set.add(p["document_id"])
        for qid in qids_chunk:
            pos_ids = pos_map[qid]["pos_ids"]
            need_ids_set.update(pos_ids)

        need_ids = list(need_ids_set)

        counts_map = fetch_counts_batch(neg_conn, need_ids)

        corpus_conn = sqlite_connect(corpus_db_path)
        texts_map = fetch_texts_batch(corpus_conn, need_ids)
        corpus_conn.close()

        bumped_all: list[str] = []

        for qid, pairs in zip(candidates["query_id"].to_list(), candidates["pairs"].to_list()):
            chosen_ids: list[str] = []
            chosen_scores: list[float] = []
            chosen_original_scores: list[float | None] = []
            chosen_tiers: list[str] = []
            bumped_local: list[str] = []
            seen_local: set[str] = set()
            pos_row = pos_map[qid]
            pos_ids = list(pos_row["pos_ids"])
            pos_scores = list(pos_row["pos_scores"])
            pos_ids_set = set(pos_ids)
            original_pos_scores = (
                list(pos_row.get("original_pos_scores") or [])
                if positive_original_score_column
                else [None] * len(pos_ids)
            )

            mined_positive_ids: list[str] = []
            mined_positive_scores: list[float] = []
            mined_positive_original_scores: list[float | None] = []
            if mine_positives and max_mined_positives > 0:
                accepted_positive_texts = [texts_map.get(pid, "") for pid in pos_ids]
                positive_candidates = sorted(pairs, key=lambda pair: pair["ranking"], reverse=True)
                for pair in positive_candidates:
                    did = pair["document_id"]
                    if did in pos_ids_set or did in mined_positive_ids:
                        continue
                    candidate_percentile = float(percentile_from_ecdf(pair["ranking"], ecdf_x, ecdf_y))
                    if candidate_percentile < u_sanity_ceiling:
                        continue
                    if (
                        candidate_percentile < u_absolute_ceiling
                        and candidate_percentile < float(pos_row["u_strongest_pos"]) * u_positive_beta
                    ):
                        continue
                    candidate_text = texts_map.get(did, "")
                    if is_near_duplicate(candidate_text, accepted_positive_texts, positive_near_duplicate_threshold):
                        stats["synthetic_positive_near_duplicate_filtered"] += 1
                        continue
                    mined_positive_ids.append(did)
                    mined_positive_scores.append(pair["ranking"])
                    mined_positive_original_scores.append(pair.get("original_negative_ranking"))
                    accepted_positive_texts.append(candidate_text)
                    if len(mined_positive_ids) >= max_mined_positives:
                        break

            pos_ids.extend(mined_positive_ids)
            pos_scores.extend(mined_positive_scores)
            pos_ids_set.update(mined_positive_ids)
            if mined_positive_ids:
                stats["queries_with_synthetic_positives"] += 1
                stats["synthetic_positives_written"] += len(mined_positive_ids)

            strict_pairs = sorted(
                (p for p in pairs if p["strict_negative"]),
                key=lambda p: p["ranking"],
                reverse=True,
            )
            fallback_pairs = sorted(
                (p for p in pairs if not p["strict_negative"]),
                key=lambda p: p["ranking"],
            )
            ordered_pairs = [(p, "strict") for p in strict_pairs]
            if backfill_policy == "relaxed":
                ordered_pairs.extend((p, "relaxed_backfill") for p in fallback_pairs)

            strict_chosen = 0
            backfill_chosen = 0
            for pair, tier in ordered_pairs:
                did = pair["document_id"]
                rnk = pair["ranking"]
                if did in seen_local:
                    stats["duplicate_candidate_filtered"] += 1
                    continue
                seen_local.add(did)
                if did in pos_ids_set:
                    stats["positive_doc_filtered"] += 1
                    continue
                if counts_map.get(did, 0) >= max_neg_reuse:
                    stats["reuse_filtered"] += 1
                    continue
                chosen_ids.append(did)
                chosen_scores.append(rnk)
                if negative_original_score_column:
                    chosen_original_scores.append(pair.get("original_negative_ranking"))
                chosen_tiers.append(tier)
                counts_map[did] = counts_map.get(did, 0) + 1
                bumped_local.append(did)
                if tier == "strict":
                    strict_chosen += 1
                else:
                    backfill_chosen += 1
                if len(chosen_ids) >= num_negatives:
                    break

            if not chosen_ids:
                stats["queries_with_zero_negatives"] += 1
                neg_count_histogram[0] += 1
                continue

            query_text = queries_dict.get(qid, "")

            pos_texts = [texts_map.get(pid, "") for pid in pos_ids]
            neg_texts = [texts_map.get(nid, "") for nid in chosen_ids]

            item = {
                "query_id": qid,
                "query": query_text,
                "pos": pos_texts,
                "neg": neg_texts,
                "pos_scores": pos_scores,
                "neg_scores": chosen_scores,
                "prompt": prompt,
                "type": dataset_type,
                "pos_id": pos_ids,
                "neg_id": chosen_ids,
                "pos_is_synthetic": [False] * (len(pos_ids) - len(mined_positive_ids))
                + [True] * len(mined_positive_ids),
                "neg_selection_tier": chosen_tiers,
            }
            if positive_original_score_column or negative_original_score_column:
                item["original_pos_scores"] = original_pos_scores + mined_positive_original_scores
            if negative_original_score_column:
                item["original_neg_scores"] = chosen_original_scores
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            bumped_all.extend(bumped_local)
            stats["queries_written"] += 1
            stats["strict_negatives_written"] += strict_chosen
            stats["backfill_negatives_written"] += backfill_chosen
            if backfill_chosen:
                stats["queries_with_backfill"] += 1
            if len(chosen_ids) < num_negatives:
                stats["queries_with_partial_negatives"] += 1
            else:
                stats["queries_with_full_negatives"] += 1
            neg_count_histogram[len(chosen_ids)] += 1

        inc_counts_batch(neg_conn, bumped_all)
        neg_conn.commit()

    out_f.close()
    neg_conn.close()
    report = {
        "target_negatives": num_negatives,
        "beta": beta,
        "u_floor": u_floor,
        "positive_score_column": positive_score_column,
        "negative_score_column": negative_score_column,
        "positive_original_score_column": positive_original_score_column,
        "negative_original_score_column": negative_original_score_column,
        "prompt": prompt,
        "type": dataset_type,
        "backfill_policy": backfill_policy,
        "low_memory_optimizations": low_memory_optimizations,
        "mine_positives": mine_positives,
        "max_mined_positives": max_mined_positives,
        "u_sanity_ceiling": u_sanity_ceiling,
        "u_absolute_ceiling": u_absolute_ceiling,
        "u_positive_beta": u_positive_beta,
        "positive_near_duplicate_threshold": positive_near_duplicate_threshold,
        "query_skip": query_skip,
        "query_limit": query_limit,
        "total_queries": total_q,
        **dict(stats),
        "negatives_per_query_histogram": {str(key): value for key, value in sorted(neg_count_histogram.items())},
    }
    write_report(report_path, report)


# --------------------------- CLI ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL for FlagEmbedding (streaming, top-K, reuse-capped, low-memory)."
    )
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"))
    parser.add_argument("--negatives_path", type=str, default=config("NEGATIVES_PATH"))
    parser.add_argument("--output_path", type=str, default=config("OUTPUT_PATH"))
    parser.add_argument("--num_negatives", type=int, default=config("NUM_NEGATIVES", cast=int, default=5))
    parser.add_argument(
        "--positive_score_column",
        type=str,
        default=config("POSITIVE_SCORE_COLUMN", default="positive_ranking"),
        help="Score column to read from relevant_path.",
    )
    parser.add_argument(
        "--negative_score_column",
        type=str,
        default=config("NEGATIVE_SCORE_COLUMN", default="ranking"),
        help="Score column to read from negatives_path.",
    )
    parser.add_argument(
        "--positive_original_score_column",
        type=str,
        default=config("POSITIVE_ORIGINAL_SCORE_COLUMN", default=None),
        help="Optional original positive score column to export as original_pos_scores.",
    )
    parser.add_argument(
        "--negative_original_score_column",
        type=str,
        default=config("NEGATIVE_ORIGINAL_SCORE_COLUMN", default=None),
        help="Optional original negative score column to export as original_neg_scores.",
    )
    parser.add_argument("--prompt", type=str, default=config("FLAG_EMBEDDING_PROMPT", default=""))
    parser.add_argument("--type", dest="dataset_type", type=str, default=config("FLAG_EMBEDDING_TYPE", default="retrieval"))

    parser.add_argument(
        "--beta",
        type=float,
        default=config("FINAL_BETA", cast=float, default=config("BETA", cast=float, default=0.01)),
        help="Warunek: u_doc <= max(beta * u_pos, u_floor) (liczone przez ECDF).",
    )
    parser.add_argument(
        "--u_floor",
        type=float,
        default=config("FINAL_U_FLOOR", cast=float, default=config("U_FLOOR", cast=float, default=0.005)),
        help="Minimalny percentyl dla negatywu.",
    )

    parser.add_argument(
        "--max_neg_reuse",
        type=int,
        default=config("MAX_NEG_REUSE", cast=int, default=1000),
        help="Maks. liczba użyć dokumentu jako negatywu w całym JSONL.",
    )

    parser.add_argument("--corpus_sqlite_path", type=str, default=config("CORPUS_SQLITE_PATH", default="corpus.sqlite"))
    parser.add_argument(
        "--negcount_sqlite_path", type=str, default=config("NEGCOUNT_SQLITE_PATH", default="negcount.sqlite")
    )

    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=config("QUERY_CHUNK_SIZE", cast=int, default=10_000),
        help="Ile query_id przetwarzać w jednym przebiegu.",
    )
    parser.add_argument(
        "--oversample_factor",
        type=int,
        default=config("OVERSAMPLE_FACTOR", cast=int, default=5),
        help="Legacy compatibility knob; relaxed backfill considers all final-scored candidates in a query chunk.",
    )
    parser.add_argument(
        "--backfill_policy",
        choices=["none", "relaxed"],
        default=config("BACKFILL_POLICY", default="relaxed"),
        help="Whether to fill missing strict negatives with the safest final-scored relaxed candidates.",
    )
    parser.add_argument("--report_path", type=str, default=config("EXPORT_REPORT_PATH", default=None))
    parser.add_argument("--low-memory-optimizations", dest="low_memory_optimizations", action="store_true")
    parser.add_argument("--no-low-memory-optimizations", dest="low_memory_optimizations", action="store_false")
    parser.set_defaults(low_memory_optimizations=config("LOW_MEMORY_OPTIMIZATIONS", cast=bool, default=False))
    parser.add_argument("--mine_positives", dest="mine_positives", action="store_true")
    parser.add_argument("--no_mine_positives", dest="mine_positives", action="store_false")
    parser.set_defaults(mine_positives=config("MINE_POSITIVES", cast=bool, default=False))
    parser.add_argument(
        "--max_mined_positives",
        type=int,
        default=config("MAX_MINED_POSITIVES", cast=int, default=1),
    )
    parser.add_argument(
        "--u_sanity_ceiling",
        type=float,
        default=config("U_SANITY_CEILING", cast=float, default=0.90),
    )
    parser.add_argument(
        "--u_absolute_ceiling",
        type=float,
        default=config("U_ABSOLUTE_CEILING", cast=float, default=0.995),
    )
    parser.add_argument(
        "--u_positive_beta",
        type=float,
        default=config("U_POSITIVE_BETA", cast=float, default=0.95),
    )
    parser.add_argument(
        "--positive_near_duplicate_threshold",
        type=float,
        default=config("POSITIVE_NEAR_DUPLICATE_THRESHOLD", cast=float, default=0.80),
    )
    parser.add_argument("--query_skip", type=int, default=config("PIPELINE_SAMPLE_SKIP", cast=int, default=0))
    sample_limit = config("PIPELINE_SAMPLE_LIMIT", cast=int, default=0)
    parser.add_argument("--query_limit", type=int, default=sample_limit if sample_limit > 0 else None)

    args = parser.parse_args()
    report_path = args.report_path or f"{args.output_path}.report.json"

    if "POLARS_MAX_THREADS" not in os.environ:
        pass

    process_negatives_streaming(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        relevant_path=args.relevant_path,
        negatives_path=args.negatives_path,
        output_path=args.output_path,
        num_negatives=args.num_negatives,
        positive_score_column=args.positive_score_column,
        negative_score_column=args.negative_score_column,
        positive_original_score_column=args.positive_original_score_column,
        negative_original_score_column=args.negative_original_score_column,
        prompt=args.prompt,
        dataset_type=args.dataset_type,
        backfill_policy=args.backfill_policy,
        report_path=report_path,
        beta=args.beta,
        u_floor=args.u_floor,
        max_neg_reuse=args.max_neg_reuse,
        corpus_sqlite_path=args.corpus_sqlite_path,
        negcount_sqlite_path=args.negcount_sqlite_path,
        query_chunk_size=args.query_chunk_size,
        oversample_factor=args.oversample_factor,
        mine_positives=args.mine_positives,
        max_mined_positives=args.max_mined_positives,
        u_sanity_ceiling=args.u_sanity_ceiling,
        u_absolute_ceiling=args.u_absolute_ceiling,
        u_positive_beta=args.u_positive_beta,
        positive_near_duplicate_threshold=args.positive_near_duplicate_threshold,
        query_skip=args.query_skip,
        query_limit=args.query_limit,
        low_memory_optimizations=args.low_memory_optimizations,
    )


if __name__ == "__main__":
    main()
