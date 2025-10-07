from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import os, re, sqlite3, hashlib, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import sys

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from rapidfuzz import fuzz
from simhash import Simhash

# -----------------------------
# Normalizacja
# -----------------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTISPACE_RE = re.compile(r'\s+')
DIGITS_RE = re.compile(r'\d+')
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = HTML_TAG_RE.sub(' ', s)
    s = URL_RE.sub(' <url> ', s)
    s = EMAIL_RE.sub(' <email> ', s)
    s = DIGITS_RE.sub(' <num> ', s)
    s = MULTISPACE_RE.sub(' ', s).strip()
    return s

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------------
# SQLite store
# -----------------------------
class HPDB:
    """
    Na dysku:
      - exact(sha1 -> kept_id) do exact-dedup
      - blocks(bucket, lenbin, id, text) – kandydaci (SimHash)
      - kept(id) – finalne kanony
    """
    def __init__(self, path: str, fresh: bool = True):
        if fresh and os.path.exists(path):
            os.remove(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        c = self.conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA temp_store=MEMORY;")
        c.execute("CREATE TABLE IF NOT EXISTS exact (sha1 TEXT PRIMARY KEY, kept_id TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS kept (id TEXT PRIMARY KEY)")
        c.execute("CREATE TABLE IF NOT EXISTS blocks (bucket INTEGER, lenbin INTEGER, id TEXT PRIMARY KEY, text TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS blocks_idx ON blocks(bucket, lenbin)")
        self.conn.commit()

    # exact
    def exact_get(self, sha1: str) -> Optional[str]:
        row = self.conn.execute("SELECT kept_id FROM exact WHERE sha1=?", (sha1,)).fetchone()
        return row[0] if row else None

    def exact_put(self, sha1: str, kept_id: str):
        self.conn.execute("INSERT OR IGNORE INTO exact(sha1, kept_id) VALUES(?,?)", (sha1, kept_id))

    # blocks
    def blocks_add_many(self, rows: List[Tuple[int,int,str,str]]):
        self.conn.executemany("INSERT OR IGNORE INTO blocks(bucket,lenbin,id,text) VALUES(?,?,?,?)", rows)

    def blocks_distinct_buckets(self) -> List[int]:
        cur = self.conn.execute("SELECT DISTINCT bucket FROM blocks ORDER BY bucket")
        return [r[0] for r in cur.fetchall()]

    def kept_add_many(self, ids: List[str]):
        self.conn.executemany("INSERT OR IGNORE INTO kept(id) VALUES(?)", ((i,) for i in ids))

    def kept_membership(self, ids: List[str]) -> List[bool]:
        if not ids:
            return []
        q = ",".join("?" for _ in ids)
        got = set(x[0] for x in self.conn.execute(f"SELECT id FROM kept WHERE id IN ({q})", ids))
        return [(i in got) for i in ids]

    def kept_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM kept").fetchone()[0]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

# -----------------------------
# Równoległe worker'y (TOP-LEVEL!)
# -----------------------------
def _prep_for_blocking(item: Tuple[str, str], bucket_bits: int, len_bin: int, ngram: int) -> Tuple[str, str, str, int, int]:
    """
    Zwraca: (doc_id, norm, sha1, bucket, lenbin)
    """
    doc_id, raw = item
    norm = normalize_text(raw or "")
    s1 = sha1_hex(norm)
    toks = norm.split()
    feats = toks if len(toks) < ngram else (" ".join(toks[i:i+ngram]) for i in range(len(toks)-ngram+1))
    h = Simhash(feats).value
    bucket = (h >> (64 - bucket_bits)) & ((1 << bucket_bits) - 1)
    lbin = len(norm) // len_bin
    return doc_id, norm, s1, bucket, lbin

def _verify_bucket_worker(db_path: str, bucket: int, verify_threshold: int) -> Tuple[List[str], List[Tuple[str,str,str,float]], List[Tuple[str,str]]]:
    """
    Czyta jeden koszyk z SQLite (read-only), robi konserwatywną deduplikację w obrębie koszyka.
    Zwraca: (kept_ids_local, dropped_rows, cluster_rows)
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("SELECT lenbin,id,text FROM blocks WHERE bucket=?", (bucket,)).fetchall()
    conn.close()
    if not rows:
        return [], [], []

    # grupuj po lenbin
    by_bin: Dict[int, List[Tuple[str,str]]] = {}
    for lbin, rid, rtext in rows:
        by_bin.setdefault(lbin, []).append((rid, rtext))

    # zachłannie: dłuższe jako kanony
    all_docs = []
    for lb, items in by_bin.items():
        for rid, rtext in items:
            all_docs.append((lb, rid, rtext, len(rtext)))
    all_docs.sort(key=lambda x: (-x[3], x[1]))

    canons: List[Tuple[str,str]] = []  # (id, text)
    kept_ids_local: List[str] = []
    dropped_rows: List[Tuple[str,str,str,float]] = []
    cluster_rows: List[Tuple[str,str]] = []

    for lb, rid, rtext, _ in all_docs:
        best_id, best_score = None, -1.0
        for cid, ctext in canons:
            # szybki strażnik długości (ogranicza porównania)
            len_ok = 0.6 <= (len(rtext)+1)/(len(ctext)+1) <= 1.67
            if not len_ok:
                continue
            sc = fuzz.token_set_ratio(rtext, ctext, score_cutoff=verify_threshold)
            if sc >= verify_threshold and sc > best_score:
                best_score, best_id = sc, cid
        if best_id is not None:
            dropped_rows.append((rid, best_id, "near", float(best_score)))
            cluster_rows.append((best_id, rid))
        else:
            canons.append((rid, rtext))
            kept_ids_local.append(rid)
            cluster_rows.append((rid, rid))
    return kept_ids_local, dropped_rows, cluster_rows

# -----------------------------
# Wynik
# -----------------------------
@dataclass
class HPResult:
    kept_count: int
    dropped_count: int
    output_prefix: str

# -----------------------------
# Główna klasa (parallel)
# -----------------------------
class HighPrecisionDeduper:
    """
    RAM-oszczędny i szybki wariant:
      - PASS A (opcjonalnie równolegle): normalizacja + SimHash blocking
      - PASS B (równolegle): walidacja koszyków (rapidfuzz)
      - PASS C: zapis kept.parquet (membership z SQLite)
    """
    def __init__(
        self,
        *,
        bucket_bits: int = 18,
        verify_threshold: int = 95,
        len_bin: int = 40,
        progress: bool = True,
        parquet_compression: str = "zstd",
        n_jobs_ingest: Optional[int] = None,   # None -> cpu_count()-1
        n_jobs_verify: Optional[int] = None,   # None -> cpu_count()-1
        ingest_parallel: bool = True,
        block_ngram: int = 3,                  # n-gramy do SimHash
    ):
        assert 8 <= bucket_bits <= 24
        self.bucket_bits = bucket_bits
        self.verify_threshold = verify_threshold
        self.len_bin = len_bin
        self.progress = progress
        self.parquet_compression = parquet_compression
        self.n_jobs_ingest = max(1, (n_jobs_ingest if n_jobs_ingest is not None else max(1, mp.cpu_count()-1)))
        self.n_jobs_verify = max(1, (n_jobs_verify if n_jobs_verify is not None else max(1, mp.cpu_count()-1)))
        self.ingest_parallel = ingest_parallel
        self.block_ngram = block_ngram

    def deduplicate_parquet(
        self,
        parquet_path: str,
        *,
        id_column: Optional[str] = None,
        text_column: str = "text",
        batch_size: int = 20_000,
        output_prefix: Optional[str] = None,
    ) -> HPResult:
        base = output_prefix or os.path.splitext(os.path.basename(parquet_path))[0]
        db_path = f"{base}_hp.sqlite"
        db = HPDB(db_path, fresh=True)

        pf = pq.ParquetFile(parquet_path)
        total = pf.metadata.num_rows if pf.metadata else None
        pbar = tqdm(total=total, unit="rows") if (self.progress and tqdm and total) else None

        # --- Parquet writer'y (strumieniowe) ---
        dropped_writer = None
        clusters_writer = None
        def write_dropped(rows: List[Tuple[str,str,str,float]]):
            nonlocal dropped_writer
            if not rows: return
            tbl = pa.Table.from_pydict({
                "dropped_id":[r[0] for r in rows],
                "kept_id":[r[1] for r in rows],
                "reason":[r[2] for r in rows],
                "score":[r[3] for r in rows],
            })
            if dropped_writer is None:
                dropped_writer = pq.ParquetWriter(f"{base}_dropped.parquet", tbl.schema, compression=self.parquet_compression)
            dropped_writer.write_table(tbl)

        def write_clusters(rows: List[Tuple[str,str]]):
            nonlocal clusters_writer
            if not rows: return
            tbl = pa.Table.from_pydict({
                "canonical_id":[r[0] for r in rows],
                "member_id":[r[1] for r in rows],
            })
            if clusters_writer is None:
                clusters_writer = pq.ParquetWriter(f"{base}_clusters.parquet", tbl.schema, compression=self.parquet_compression)
            clusters_writer.write_table(tbl)

        # ===========================
        # PASS A: Ingest (exact + blocking)
        # ===========================
        print(f"[HP] PASS A: ingest+block -> {db_path} | n_jobs_ingest={self.n_jobs_ingest if self.ingest_parallel else 1}")
        global_row = 0
        prep_partial = partial(_prep_for_blocking, bucket_bits=self.bucket_bits, len_bin=self.len_bin, ngram=self.block_ngram)

        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column] + ([id_column] if id_column else [])):
            df = pa.Table.from_batches([batch]).to_pandas()
            if id_column is None:
                df["_id"] = [f"{base}-{i}" for i in range(global_row, global_row+len(df))]
                use_id = "_id"
            else:
                use_id = id_column
                df[use_id] = df[use_id].astype(str)

            items = list(zip(df[use_id].tolist(), df[text_column].tolist()))
            if self.ingest_parallel and self.n_jobs_ingest > 1:
                chunk = max(1000, len(items)//(self.n_jobs_ingest*4) or 1)
                with ProcessPoolExecutor(max_workers=self.n_jobs_ingest) as ex:
                    prepped = list(ex.map(prep_partial, items, chunksize=chunk))
            else:
                prepped = [prep_partial(it) for it in items]

            dropped_rows, block_rows, cluster_rows = [], [], []
            # exact-dedup i rejestracja bloków — w procesie głównym
            for doc_id, norm, s1, bucket, lbin in prepped:
                kept = db.exact_get(s1)
                if kept is not None:
                    dropped_rows.append((doc_id, kept, "exact", 1.0))
                    cluster_rows.append((kept, doc_id))
                else:
                    db.exact_put(s1, doc_id)
                    block_rows.append((bucket, lbin, doc_id, norm))

            if block_rows:
                db.blocks_add_many(block_rows)
            if dropped_rows:
                write_dropped(dropped_rows)
                write_clusters(cluster_rows)

            global_row += len(df)
            if pbar: pbar.update(len(df))

        if pbar: pbar.close()
        db.commit()  # PASS B widzi wszystko

        # ===========================
        # PASS B: Walidacja w koszykach
        # ===========================
        print(f"[HP] PASS B: verify buckets (n_jobs_verify={self.n_jobs_verify}, threshold={self.verify_threshold})")
        buckets = db.blocks_distinct_buckets()
        pbar2 = tqdm(total=len(buckets), unit="bucket") if (self.progress and tqdm) else None

        worker = partial(_verify_bucket_worker, db_path, verify_threshold=self.verify_threshold)
        # strumieniuj wyniki z procesów i od razu zapisuj
        chunk = max(50, len(buckets)//(self.n_jobs_verify*10) or 1)
        with ProcessPoolExecutor(max_workers=self.n_jobs_verify) as ex:
            for kept_ids_local, dropped_rows, cluster_rows in ex.map(worker, buckets, chunksize=chunk):
                if kept_ids_local:
                    db.kept_add_many(kept_ids_local)
                if dropped_rows:
                    write_dropped(dropped_rows)
                if cluster_rows:
                    write_clusters(cluster_rows)
                if pbar2: pbar2.update(1)

        if pbar2: pbar2.close()
        kept_count = db.kept_count()

        if dropped_writer: dropped_writer.close()
        if clusters_writer: clusters_writer.close()

        # ===========================
        # PASS C: Zapis kept.parquet
        # ===========================
        print(f"[HP] PASS C: writing {base}_kept.parquet ...")
        kept_writer = None
        total = pf.metadata.num_rows if pf.metadata else None
        pbar3 = tqdm(total=total, unit="rows") if (self.progress and tqdm and total) else None
        global_row = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            df = pa.Table.from_batches([batch]).to_pandas()
            if id_column is None:
                df["_id"] = [f"{base}-{i}" for i in range(global_row, global_row+len(df))]
                use_id = "_id"
            else:
                use_id = id_column
                df[use_id] = df[use_id].astype(str)
            ids = df[use_id].tolist()
            mask = db.kept_membership(ids)
            if any(mask):
                kept_df = df[pd.Series(mask).values]
                tbl = pa.Table.from_pandas(kept_df)
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(f"{base}_kept.parquet", tbl.schema, compression=self.parquet_compression)
                kept_writer.write_table(tbl)
            global_row += len(df)
            if pbar3: pbar3.update(len(df))
        if kept_writer: kept_writer.close()
        if pbar3: pbar3.close()

        # policz dropped (opcjonalnie)
        dropped_count = 0
        try:
            dpf = pq.ParquetFile(f"{base}_dropped.parquet")
            dropped_count = sum(b.num_rows for b in dpf.iter_batches())
        except Exception:
            pass

        db.close()
        print(f"[HP] DONE. kept={kept_count} dropped≈{dropped_count} | files: {base}_kept.parquet, {base}_dropped.parquet, {base}_clusters.parquet, {base}_hp.sqlite")
        return HPResult(kept_count=kept_count, dropped_count=dropped_count, output_prefix=base)

# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-precision dedup + filtrowanie powiązanych plików Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Parquet I/O — domyślne ścieżki
    p.add_argument("--corpus", default="corpus.parquet",
                   help="Wejściowy Parquet z korpusem do deduplikacji.")
    p.add_argument("--relevant", default="relevant.parquet",
                   help="Wejściowy Parquet z kolumną 'document_id' do przefiltrowania.")
    p.add_argument("--relevant-out", default="relevant_hp_kept.parquet",
                   help="Wyjściowy Parquet z przefiltrowanymi rekordami.")
    p.add_argument("--queries", default="queries.parquet",
                   help="Wejściowy Parquet z zapytaniami (kolumna 'id').")
    p.add_argument("--queries-out", default="queries_hp_kept.parquet",
                   help="Wyjściowy Parquet z przefiltrowanymi zapytaniami.")
    p.add_argument("--corpus-kept", default="corpus_hp_kept.parquet",
                   help="Opcjonalnie: ścieżka do już istniejącego *_kept.parquet. "
                        "Jeśli plik nie istnieje, zostanie wytworzony na podstawie --corpus/--output-prefix.")

    # Kolumny i batch
    p.add_argument("--id-column", default="id", help="Nazwa kolumny ID w korpusie.")
    p.add_argument("--text-column", default="text", help="Nazwa kolumny tekstowej w korpusie.")
    p.add_argument("--batch-size", type=int, default=20_000, help="Rozmiar batcha przy czytaniu Parquet.")

    # Parametry deduplikacji
    p.add_argument("--output-prefix", default="corpus_hp",
                   help="Prefix nazw plików wynikowych deduplikacji (kept/dropped/clusters/sqlite).")
    p.add_argument("--bucket-bits", type=int, default=18, help="Liczba bitów dla bucketowania SimHash.")
    p.add_argument("--verify-threshold", type=int, default=60, help="Próg podobieństwa (RapidFuzz).")
    p.add_argument("--len-bin", type=int, default=40, help="Długość binu dla strażnika długości.")
    p.add_argument("--n-jobs-ingest", type=int, help="Liczba procesów dla PASS A. Domyślnie cpu_count()-1.")
    p.add_argument("--n-jobs-verify", type=int, help="Liczba procesów dla PASS B. Domyślnie cpu_count()-1.")
    p.add_argument("--no-ingest-parallel", action="store_true", help="Wyłącz równoległe przetwarzanie w PASS A.")
    p.add_argument("--no-progress", action="store_true", help="Ukryj paski postępu.")

    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # 1) Dedup korpusu
    deduper = HighPrecisionDeduper(
        bucket_bits=args.bucket_bits,
        verify_threshold=args.verify_threshold,
        len_bin=args.len_bin,
        n_jobs_ingest=args.n_jobs_ingest,
        n_jobs_verify=args.n_jobs_verify,
        ingest_parallel=not args.no_ingest_parallel,
        progress=not args.no_progress,
    )

    res = deduper.deduplicate_parquet(
        args.corpus,
        id_column=args.id_column,
        text_column=args.text_column,
        batch_size=args.batch_size,
        output_prefix=args.output_prefix,
    )

    # Wyznacz ścieżkę do kept.parquet: albo podana, albo wynik deduplikacji
    corpus_kept_path = args.corpus_kept if args.corpus_kept else f"{res.output_prefix}_kept.parquet"

    # 2) Filtrowanie relevant i queries w oparciu o kept
    print(f"[HP] Filtering relevant/queries przy użyciu: {corpus_kept_path}")
    relevant_path = args.relevant
    output_path = args.relevant_out
    queries_path = args.queries
    queries_out_path = args.queries_out

    # Wczytanie plików
    relevant = pd.read_parquet(relevant_path, engine="pyarrow")
    corpus_hp_kept = pd.read_parquet(corpus_kept_path, engine="pyarrow")

    # Upewnij się, że kolumna id w kept jest typu porównywalnego z document_id
    if "id" not in corpus_hp_kept.columns:
        raise KeyError(f"Kolumna 'id' nieznaleziona w {corpus_kept_path}.")
    # rzutowanie ostrożne: spróbuj dopasować do typu document_id
    if "document_id" not in relevant.columns:
        raise KeyError(f"Kolumna 'document_id' nieznaleziona w {relevant_path}.")

    # Dopasuj typy
    if pd.api.types.is_integer_dtype(relevant["document_id"]):
        corpus_hp_kept["id"] = pd.to_numeric(corpus_hp_kept["id"], errors="coerce").astype("Int64")
    else:
        corpus_hp_kept["id"] = corpus_hp_kept["id"].astype(str)
        relevant["document_id"] = relevant["document_id"].astype(str)

    # Filtrowanie wierszy – zostaw tylko te, które są w kept
    filtered = relevant[relevant["document_id"].isin(corpus_hp_kept["id"])]

    # Zapis
    filtered.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[HP] Zapisano: {output_path}")

    # Queries
    queries = pd.read_parquet(queries_path, engine="pyarrow")
    if "id" not in queries.columns:
        raise KeyError(f"Kolumna 'id' nieznaleziona w {queries_path}.")
    if "query_id" not in filtered.columns:
        raise KeyError(f"Kolumna 'query_id' nieznaleziona w {output_path} (wynik 'relevant').")

    # Dopasuj typy
    if pd.api.types.is_integer_dtype(filtered["query_id"]):
        queries["id"] = pd.to_numeric(queries["id"], errors="coerce").astype("Int64")
    else:
        queries["id"] = queries["id"].astype(str)
        filtered["query_id"] = filtered["query_id"].astype(str)

    queries_hp_kept = queries[queries["id"].isin(filtered["query_id"])].copy()
    queries_hp_kept.to_parquet(queries_out_path, index=False, engine="pyarrow")
    print(f"[HP] Zapisano: {queries_out_path}")

    print("[HP] Gotowe.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
