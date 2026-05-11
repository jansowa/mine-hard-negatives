import os

os.environ.setdefault("POLARS_MAX_THREADS", "12")

import argparse
import json
import sqlite3
from collections import Counter

import numpy as np
import polars as pl
from decouple import config
from tqdm.auto import tqdm

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


# --------------------------- Build corpus SQLite ---------------------------


def build_or_load_corpus_sqlite(
    corpus_path: str, corpus_sqlite_path: str, id_col="id", text_col="text", block_rows: int = 1_000_000
):
    conn = sqlite_connect(corpus_sqlite_path)
    ensure_corpus_db(conn)

    cur = conn.execute("SELECT COUNT(1) FROM corpus;")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    ds = pl.scan_parquet(corpus_path).select([pl.col(id_col).cast(pl.Utf8), pl.col(text_col).cast(pl.Utf8)])

    df = ds.collect()

    n = df.height
    with tqdm(total=n, desc="Budowanie bazy korpusu (SQLite)", unit="rows") as pbar:
        for start in range(0, n, block_rows):
            end = min(start + block_rows, n)
            chunk = df.slice(start, end - start)
            rows = zip(chunk[id_col].to_list(), chunk[text_col].to_list())
            bulk_upsert_corpus(conn, rows)
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
    backfill_policy: str = "relaxed",
    report_path: str | None = None,
):
    if not positive_score_column:
        raise ValueError("positive_score_column must not be empty")
    if not negative_score_column:
        raise ValueError("negative_score_column must not be empty")
    if backfill_policy not in {"none", "relaxed"}:
        raise ValueError("backfill_policy must be one of: none, relaxed")

    try:
        pl.enable_string_cache()
    except AttributeError:
        pass

    build_or_load_corpus_sqlite(
        corpus_path=corpus_path,
        corpus_sqlite_path=corpus_sqlite_path,
        block_rows=1_000_000,
    )

    neg_conn = sqlite_connect(negcount_sqlite_path)
    ensure_negcount_db(neg_conn)

    qdf = pl.read_parquet(queries_path, columns=["id", "text"]).with_columns(
        pl.col("id").cast(pl.Utf8), pl.col("text").cast(pl.Utf8)
    )
    queries_dict = dict(zip(qdf["id"].to_list(), qdf["text"].to_list()))
    del qdf

    rel = (
        pl.scan_parquet(relevant_path)
        .select(
            [
                pl.col("query_id"),
                pl.col("document_id"),
                pl.col(positive_score_column).alias("positive_ranking"),
            ]
        )
        .with_columns(
            pl.col("query_id").cast(pl.Utf8),
            pl.col("document_id").cast(pl.Utf8),
            pl.col("positive_ranking").cast(pl.Float64),
        )
    )

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

    pos_grouped = (
        rel.group_by("query_id")
        .agg(
            pl.col("document_id").implode().alias("pos_ids"),
            pl.col("positive_ranking").implode().alias("pos_scores"),
            pl.col("positive_ranking").min().alias("min_score"),
        )
        .with_columns(pl.col("min_score").cast(pl.Float64))
        .collect()
    )

    min_scores = pos_grouped["min_score"].to_numpy()
    u_pos = percentile_from_ecdf(min_scores, ecdf_x, ecdf_y)
    thr1 = inv_percentile_from_ecdf(beta * u_pos, ecdf_x, ecdf_y)
    thr2 = inv_percentile_from_ecdf(u_floor, ecdf_x, ecdf_y)
    thresholds = np.maximum(thr1, thr2)

    thr_df = pl.DataFrame(
        {
            "query_id": pos_grouped["query_id"],
            "pos_ids": pos_grouped["pos_ids"],
            "pos_scores": pos_grouped["pos_scores"],
            "min_score": pos_grouped["min_score"],
            "u_pos": pl.Series(u_pos),
            "threshold_rank": pl.Series(thresholds),
        }
    )

    pos_map: dict[str, tuple[list[str], list[float]]] = {
        qid: (row["pos_ids"], row["pos_scores"]) for qid, row in zip(thr_df["query_id"], thr_df.iter_rows(named=True))
    }

    neg_scan = pl.scan_parquet(negatives_path).select(
        [
            pl.col("query_id").cast(pl.Utf8),
            pl.col("document_id").cast(pl.Utf8),
            pl.col(negative_score_column).cast(pl.Float64).alias("ranking"),
        ]
    )

    all_qids = thr_df["query_id"].to_list()
    total_q = len(all_qids)

    out_f = open(output_path, "w", encoding="utf-8")
    corpus_db_path = os.path.abspath(corpus_sqlite_path)
    stats: Counter[str] = Counter()
    neg_count_histogram: Counter[int] = Counter()

    for start in tqdm(range(0, total_q, query_chunk_size), desc="Negatives streaming", unit="q_chunk"):
        end = min(start + query_chunk_size, total_q)
        qids_chunk = all_qids[start:end]

        thr_chunk = thr_df.slice(start, end - start).select(["query_id", "threshold_rank"])

        candidates = (
            neg_scan.join(thr_chunk.lazy(), on="query_id", how="inner")
            .with_columns((pl.col("ranking") <= pl.col("threshold_rank")).alias("strict_negative"))
            .group_by("query_id")
            .agg(pl.struct(["document_id", "ranking", "strict_negative"]).alias("pairs"))
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
            pos_ids, _ = pos_map[qid]
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
            chosen_tiers: list[str] = []
            bumped_local: list[str] = []
            seen_local: set[str] = set()
            pos_ids, pos_scores = pos_map[qid]
            pos_ids_set = set(pos_ids)

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
                "query": query_text,
                "pos": pos_texts,
                "neg": neg_texts,
                "pos_scores": pos_scores,
                "neg_scores": chosen_scores,
                "pos_id": pos_ids,
                "neg_id": chosen_ids,
                "neg_selection_tier": chosen_tiers,
            }
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
        "backfill_policy": backfill_policy,
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
        backfill_policy=args.backfill_policy,
        report_path=report_path,
        beta=args.beta,
        u_floor=args.u_floor,
        max_neg_reuse=args.max_neg_reuse,
        corpus_sqlite_path=args.corpus_sqlite_path,
        negcount_sqlite_path=args.negcount_sqlite_path,
        query_chunk_size=args.query_chunk_size,
        oversample_factor=args.oversample_factor,
    )


if __name__ == "__main__":
    main()
