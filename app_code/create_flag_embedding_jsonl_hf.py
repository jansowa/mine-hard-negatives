#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from heapq import heappush, heappushpop
from typing import Dict, List, Tuple, Optional, Iterable

from datasets import load_dataset
from decouple import config
from tqdm import tqdm


# ----------------------------- utils: sqlite -------------------------------- #

def open_sqlite(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # przyspieszenia I/O (bezpieczeństwo OK dla local tmp)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=30000000000;")
    conn.commit()
    return conn


def create_sqlite_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_texts(
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_texts(
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS needed_docs(
            id INTEGER PRIMARY KEY
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_needed_docs ON needed_docs(id);")
    conn.commit()


def insert_many(conn: sqlite3.Connection, sql: str, rows: Iterable[Tuple], batch: int = 1000) -> None:
    cur = conn.cursor()
    buf = []
    for r in rows:
        buf.append(r)
        if len(buf) >= batch:
            cur.executemany(sql, buf)
            conn.commit()
            buf.clear()
    if buf:
        cur.executemany(sql, buf)
        conn.commit()


def select_texts(conn: sqlite3.Connection, table: str, ids: List[int]) -> Dict[int, str]:
    cur = conn.cursor()
    out = {}
    for i in range(0, len(ids), 1000):
        chunk = ids[i:i+1000]
        placeholders = ",".join(["?"] * len(chunk))
        cur.execute(f"SELECT id, text FROM {table} WHERE id IN ({placeholders})", chunk)
        for did, txt in cur.fetchall():
            out[int(did)] = txt
    return out


def fetch_one_text(conn: sqlite3.Connection, table: str, doc_id: int) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(f"SELECT text FROM {table} WHERE id=?", (int(doc_id),))
    row = cur.fetchone()
    return row[0] if row else None


# ----------------------- główna logika low-RAM ------------------------------ #

def process_negatives_hf_lowram(
    dataset_id: str,
    output_path: str,
    num_negatives: int,
    negatives_multiplication_threshold: Optional[float],
    negatives_subtraction_threshold: Optional[float],
    hf_token: Optional[str],
    split: str,
    sqlite_path: str,
    membership_backend: str = "set",
    negatives_limit: Optional[int] = None,   # NEW
    negatives_skip: int = 0,                 # NEW
    negatives_glob: Optional[str] = None,
) -> None:

    if negatives_subtraction_threshold is not None and negatives_multiplication_threshold is not None:
        raise ValueError("You have to choose between subtraction and multiplication threshold.")
    if negatives_subtraction_threshold is None and negatives_multiplication_threshold is None:
        raise ValueError("You have to fill in the negatives threshold")
    if membership_backend not in {"set", "sqlite"}:
        raise ValueError("--membership_backend must be 'set' or 'sqlite'")

    # 0) SQLite setup
    conn = open_sqlite(sqlite_path)
    create_sqlite_schema(conn)

    # 1) Pozytywy + progi (relevant_with_score jest mały — ale i tak streamujemy)
    print("Krok 1/4: wczytuję positive pairs i liczę progi...")
    print(f"{hf_token=}")
    print(f"{dataset_id=}")
    relevant = load_dataset(dataset_id, "relevant_with_score", split=split, streaming=True, token=hf_token)

    pos_ids: Dict[int, List[int]] = defaultdict(list)
    pos_scores: Dict[int, List[float]] = defaultdict(list)
    min_pos_score: Dict[int, float] = {}

    cnt_rel = 0
    for ex in tqdm(relevant, desc="relevant_with_score (stream)"):
        qid = int(ex["query_id"])
        did = int(ex["document_id"])
        sc = float(ex["positive_ranking"])
        pos_ids[qid].append(did)
        pos_scores[qid].append(sc)
        m = min_pos_score.get(qid)
        if m is None or sc < m:
            min_pos_score[qid] = sc
        cnt_rel += 1

    print(f"  Zebrano {cnt_rel} połączeń pozytywnych, unikalnych qid: {len(min_pos_score):,}")

    thresholds: Dict[int, float] = {}
    if negatives_multiplication_threshold is not None:
        for qid, m in min_pos_score.items():
            thresholds[qid] = m * negatives_multiplication_threshold
    else:
        for qid, m in min_pos_score.items():
            thresholds[qid] = m - negatives_subtraction_threshold  # type: ignore[arg-type]

    needed_qids = set(pos_ids.keys())

    # 2) Query teksty → SQLite (tylko potrzebne qid)
    print("Krok 2/4: zapisuję teksty zapytań (tylko potrzebne qid) do SQLite...")
    queries = load_dataset(dataset_id, "queries", split=split, streaming=True, token=hf_token)
    cur = conn.cursor()
    buf = []
    for ex in tqdm(queries, desc="queries (stream)"):
        qid = int(ex["id"])
        if qid in needed_qids:
            buf.append((qid, ex["text"]))
            if len(buf) >= 1000:
                cur.executemany("INSERT OR REPLACE INTO query_texts(id, text) VALUES (?, ?)", buf)
                conn.commit()
                buf.clear()
    if buf:
        cur.executemany("INSERT OR REPLACE INTO query_texts(id, text) VALUES (?, ?)", buf)
        conn.commit()

    # 3) Negatywy (stream) — top-K per qid wg progu
    print("Krok 3/4: wybieram top-K negatywów per qid w trybie stream...")
    data_files_arg = None
    if negatives_glob:
        data_files_arg = {"train": negatives_glob}

    stream_cache_dir = os.getenv("HF_STREAM_CACHE", "/cache/hf/stream")
    storage_options = {"simplecache": {"cache_storage": stream_cache_dir}}
    negatives = load_dataset(
        dataset_id,
        "negatives",
        split=split,
        streaming=True,
        token=hf_token,
        data_files=data_files_arg,  # << ograniczamy się do wskazanych shardów
        storage_options=storage_options,
    )

    # Dla RAMu: min-heap (score, doc_id) o rozmiarze <= K na qid
    neg_heaps: Dict[int, List[Tuple[float, int]]] = {}

    processed = 0
    skipped = 0
    desc = f"negatives (stream, limit={negatives_limit or 'ALL'}, skip={negatives_skip})"

    for ex in tqdm(negatives, desc=desc):
        # skip pierwszych N wierszy (bez liczenia do limitu)
        if skipped < negatives_skip:
            skipped += 1
            continue

        # licznik przetworzonych (pod limit)
        if negatives_limit is not None and processed >= negatives_limit:
            break
        processed += 1

        qid = int(ex["query_id"])
        thr = thresholds.get(qid)
        if thr is None:
            continue
        score = float(ex["ranking"])
        if score >= thr:
            continue
        did = int(ex["document_id"])
        heap = neg_heaps.get(qid)
        pair = (score, did)
        if heap is None:
            heap = []
            neg_heaps[qid] = heap
        if len(heap) < num_negatives:
            heappush(heap, pair)
        else:
            if score > heap[0][0]:
                heappushpop(heap, pair)

    print(f"  negatives: skipped={skipped:,}, processed={processed:,}")

    # Zamień heapy na posortowane listy (malejąco po score)
    neg_ids_by_qid: Dict[int, List[int]] = {}
    neg_scores_by_qid: Dict[int, List[float]] = {}
    for qid, heap in neg_heaps.items():
        heap.sort(key=lambda x: x[0], reverse=True)
        neg_scores_by_qid[qid] = [float(s) for s, _ in heap]
        neg_ids_by_qid[qid] = [int(d) for _, d in heap]
    neg_heaps.clear()

    # 4) Lista potrzebnych dokumentów (pozytywne + wybrane negatywne)
    print("Krok 4/4: buduję listę potrzebnych document_id...")
    if membership_backend == "set":
        needed_doc_ids = set()  # szybki membership, kosztem RAM
        for ids in pos_ids.values():
            needed_doc_ids.update(ids)
        for ids in neg_ids_by_qid.values():
            needed_doc_ids.update(ids)
        print(f"  Potrzebnych dokumentów: ~{len(needed_doc_ids):,}")
    else:
        # SQLite membership (mało RAM, wolniejsze)
        print("  Używam SQLite do przechowywania needed_doc_ids (mało RAM).")
        cur = conn.cursor()
        # pozytywne
        rows = ((int(did),) for ids in pos_ids.values() for did in ids)
        insert_many(conn, "INSERT OR IGNORE INTO needed_docs(id) VALUES (?)", rows, batch=5000)
        # negatywne
        rows = ((int(did),) for ids in neg_ids_by_qid.values() for did in ids)
        insert_many(conn, "INSERT OR IGNORE INTO needed_docs(id) VALUES (?)", rows, batch=5000)

    # 5) Jednoprzebiegowe zebranie tekstów z corpus tylko dla potrzebnych id → SQLite
    print("Przechodzę corpus (stream) i zapisuję tylko potrzebne teksty do SQLite...")
    corpus = load_dataset(dataset_id, "corpus", split=split, streaming=True, token=hf_token, storage_options=storage_options)
    cur = conn.cursor()
    buf = []
    checked = 0
    hit = 0

    if membership_backend == "set":
        for ex in tqdm(corpus, desc="corpus (stream)"):
            did = int(ex["id"])
            if did in needed_doc_ids:  # szybkie O(1)
                buf.append((did, ex["text"]))
                hit += 1
                if len(buf) >= 1000:
                    cur.executemany("INSERT OR REPLACE INTO doc_texts(id, text) VALUES (?, ?)", buf)
                    conn.commit()
                    buf.clear()
            checked += 1
    else:
        # SQL membership check (index na needed_docs)
        for ex in tqdm(corpus, desc="corpus (stream)"):
            did = int(ex["id"])
            cur.execute("SELECT 1 FROM needed_docs WHERE id=?", (did,))
            if cur.fetchone():
                buf.append((did, ex["text"]))
                hit += 1
                if len(buf) >= 1000:
                    cur.executemany("INSERT OR REPLACE INTO doc_texts(id, text) VALUES (?, ?)", buf)
                    conn.commit()
                    buf.clear()
            checked += 1

    if buf:
        cur.executemany("INSERT OR REPLACE INTO doc_texts(id, text) VALUES (?, ?)", buf)
        conn.commit()
        buf.clear()

    print(f"  Sprawdzono wierszy corpus: ~{checked:,}, trafień (zapisanych): ~{hit:,}")

    # 6) Generowanie wyjścia JSONL — pobieramy teksty z SQLite na żądanie
    print(f"Generuję wynik: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for qid in tqdm(pos_ids.keys(), desc="write jsonl"):
            # query text
            qtext = fetch_one_text(conn, "query_texts", qid) or ""

            # pozytywy
            p_ids = pos_ids[qid]
            p_texts = []
            for did in p_ids:
                t = fetch_one_text(conn, "doc_texts", did)
                if t is not None:
                    p_texts.append(t)
                else:
                    p_texts.append("")  # albo pomiń

            # negatywy (mogą nie istnieć)
            n_ids = neg_ids_by_qid.get(qid, [])
            n_scores = neg_scores_by_qid.get(qid, [])
            n_texts = []
            for did in n_ids:
                t = fetch_one_text(conn, "doc_texts", did)
                if t is not None:
                    n_texts.append(t)
                else:
                    n_texts.append("")

            item = {
                "query": qtext,
                "pos": p_texts,
                "neg": n_texts,
                "pos_scores": pos_scores[qid],
                "neg_scores": n_scores,
                "pos_id": p_ids,
                "neg_id": n_ids,
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Zapisano plik: {output_path}")
    conn.close()


# ------------------------------- CLI --------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL with positives/negatives/scores directly from a HuggingFace dataset (low-RAM)."
    )
    parser.add_argument("--hf_dataset", type=str, default=config("HF_DATASET", default="minehard/negatives"))
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", config("HF_TOKEN", default="")))
    parser.add_argument("--split", type=str, default=config("HF_SPLIT", default="train"))
    parser.add_argument("--output_path", type=str, default=config("OUTPUT_PATH", default="output.jsonl"))
    parser.add_argument("--num_negatives", type=int, default=config("NUM_NEGATIVES", cast=int, default=5))
    parser.add_argument(
        "--negatives_limit",
        type=int,
        default=None,
        help="Maksymalna liczba wierszy z 'negatives' do przetworzenia (np. 100000). Domyślnie: ALL.",
    )
    parser.add_argument(
        "--negatives_skip",
        type=int,
        default=0,
        help="Liczba początkowych wierszy 'negatives' do pominięcia przed liczeniem limitu. Domyślnie: 0.",
    )
    parser.add_argument(
        "--negatives_glob",
        type=str,
        default=None,
        help="Glob dla plików 'negatives' (np. 'negatives/train-00000-of-*')."
    )

    # thresholds
    neg_sub_str = config("NEGATIVES_SUBTRACTION_THRESHOLD", default=None)
    neg_mul_str = config("NEGATIVES_MULTIPLICATION_THRESHOLD", default=None)
    neg_sub = float(neg_sub_str) if neg_sub_str else None
    neg_mul = float(neg_mul_str) if neg_mul_str else None

    # override by CLI if present
    parser.add_argument("--negatives_multiplication_threshold", required=False, type=float,
                        default=neg_mul, help="percent of min positive (e.g. 0.9)")
    parser.add_argument("--negatives_subtraction_threshold", required=False, type=float,
                        default=neg_sub, help="subtract from min positive (e.g. 0.05)")

    parser.add_argument("--sqlite_path", type=str, default=config("SQLITE_PATH", default="hf_lowram/cache.db"),
                        help="Path to local SQLite file for texts & membership")
    parser.add_argument("--membership_backend", choices=["set", "sqlite"],
                        default=config("MEMBERSHIP_BACKEND", default="set"),
                        help="'set' (faster, more RAM) or 'sqlite' (slower, low RAM)")

    args = parser.parse_args()

    token = args.hf_token if args.hf_token else None

    process_negatives_hf_lowram(
        dataset_id=args.hf_dataset,
        output_path=args.output_path,
        num_negatives=args.num_negatives,
        negatives_multiplication_threshold=args.negatives_multiplication_threshold,
        negatives_subtraction_threshold=args.negatives_subtraction_threshold,
        hf_token=token,
        split=args.split,
        sqlite_path=args.sqlite_path,
        membership_backend=args.membership_backend,
        negatives_limit=args.negatives_limit,  # NEW
        negatives_skip=args.negatives_skip,  # NEW
        negatives_glob=args.negatives_glob,
    )


if __name__ == "__main__":
    main()


# python create_flag_embedding_jsonl_hf.py \
#   --hf_dataset minehard/negatives \
#   --split train \
#   --output_path out.jsonl \
#   --num_negatives 5 \
#   --negatives_subtraction_threshold 5.0 \
#   --membership_backend sqlite \
#   --sqlite_path ./hf_lowram/cache.db \
#   --negatives_glob "negatives/train-00000-of-*" \
#   --negatives_limit 10000
#
#
#
#
#
# python create_flag_embedding_jsonl_hf.py \
#   --hf_dataset minehard/negatives \
#   --split train \
#   --output_path out_7.jsonl \
#   --num_negatives 10 \
#   --negatives_subtraction_threshold 7.0 \
#   --membership_backend sqlite \
#   --sqlite_path ./hf_lowram/cache.db