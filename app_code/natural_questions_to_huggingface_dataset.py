import argparse
import gzip
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from decouple import config
from huggingface_hub import hf_hub_download

BATCH_SIZE = 10_000
DEFAULT_INPUT_PATH = str(Path(__file__).resolve().parent / "data" / "natural-questions.jsonl")
QUERY_ID_RE = re.compile(r"(\d+)$")


@dataclass
class NaturalQuestionsStats:
    processed_rows: int = 0
    written_queries: int = 0
    written_corpus_docs: int = 0
    written_relevant_pairs: int = 0
    skipped_bad_json: int = 0
    skipped_bad_id: int = 0
    skipped_empty_query: int = 0
    skipped_bad_texts_or_labels: int = 0


def write_batch(writer: pq.ParquetWriter, batch: list[dict], schema: pa.Schema) -> None:
    if not batch:
        return
    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    batch.clear()


def extract_trailing_query_id(raw_id: object) -> Optional[str]:
    match = QUERY_ID_RE.search(str(raw_id or ""))
    if not match:
        return None
    return match.group(1)


def reservoir_sample_msmarco_docs(
    docs: Iterable[dict],
    sample_size: int,
    used_ids: set[str],
    rng: random.Random,
) -> tuple[list[dict], int, int]:
    if sample_size <= 0:
        return [], 0, 0

    reservoir: list[dict] = []
    eligible_seen = 0
    skipped_used = 0

    for item in docs:
        doc_id = str(item.get("_id", ""))
        text = item.get("text")

        if not doc_id or not text:
            continue
        if doc_id in used_ids:
            skipped_used += 1
            continue

        row = {"id": doc_id, "text": text}
        eligible_seen += 1

        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue

        replace_idx = rng.randint(0, eligible_seen - 1)
        if replace_idx < sample_size:
            reservoir[replace_idx] = row

    return reservoir, eligible_seen, skipped_used


def download_msmarco_pl_corpus(hf_token: Optional[str]) -> str:
    try:
        return hf_hub_download(
            repo_id="clarin-knext/msmarco-pl",
            repo_type="dataset",
            filename="corpus.jsonl.gz",
            token=hf_token,
        )
    except Exception:
        try:
            return hf_hub_download(
                repo_id="clarin-knext/msmarco-pl",
                repo_type="dataset",
                filename="corpus.jsonl.gz",
                token=hf_token,
                local_files_only=True,
            )
        except Exception as cache_exc:
            raise RuntimeError(
                "Could not download clarin-knext/msmarco-pl corpus.jsonl.gz and no cached copy "
                "was available. Check network access to Hugging Face and try again."
            ) from cache_exc


def iter_jsonl_gz(path: str) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            yield json.loads(line)


def write_natural_questions_parquets(
    input_path: str,
    queries_writer: pq.ParquetWriter,
    corpus_writer: pq.ParquetWriter,
    relevant_writer: pq.ParquetWriter,
    queries_schema: pa.Schema,
    corpus_schema: pa.Schema,
    relevant_schema: pa.Schema,
    nq_doc_id_start: int,
    batch_size: int,
) -> NaturalQuestionsStats:
    stats = NaturalQuestionsStats()
    next_doc_id = nq_doc_id_start

    queries_batch: list[dict] = []
    corpus_batch: list[dict] = []
    relevant_batch: list[dict] = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            stats.processed_rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped_bad_json += 1
                print(f"  Skipping line {line_number}: invalid JSON.")
                continue

            query_id = extract_trailing_query_id(item.get("id"))
            if query_id is None:
                stats.skipped_bad_id += 1
                continue

            query_text = (item.get("query") or "").strip()
            if not query_text:
                stats.skipped_empty_query += 1
                continue

            texts = item.get("texts")
            labels = item.get("labels")
            if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) != len(labels):
                stats.skipped_bad_texts_or_labels += 1
                continue

            queries_batch.append({"id": query_id, "text": query_text})
            stats.written_queries += 1

            for text, label in zip(texts, labels):
                doc_id = str(next_doc_id)
                next_doc_id += 1

                corpus_batch.append({"id": doc_id, "text": str(text or "")})
                stats.written_corpus_docs += 1

                if label == 1:
                    relevant_batch.append({"query_id": query_id, "document_id": doc_id})
                    stats.written_relevant_pairs += 1

                if len(corpus_batch) >= batch_size:
                    write_batch(corpus_writer, corpus_batch, corpus_schema)
                    print(f"  Written {stats.written_corpus_docs} Natural Questions corpus docs...")

                if len(relevant_batch) >= batch_size:
                    write_batch(relevant_writer, relevant_batch, relevant_schema)
                    print(f"  Written {stats.written_relevant_pairs} relevant pairs...")

            if len(queries_batch) >= batch_size:
                write_batch(queries_writer, queries_batch, queries_schema)
                print(f"  Written {stats.written_queries} queries...")

    write_batch(queries_writer, queries_batch, queries_schema)
    write_batch(corpus_writer, corpus_batch, corpus_schema)
    write_batch(relevant_writer, relevant_batch, relevant_schema)

    return stats


def load_used_corpus_ids(dataset_name: str, hf_token: Optional[str]) -> set[str]:
    try:
        ds = load_dataset(dataset_name, data_dir="corpus", split="train", token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the private used-corpus dataset. Set a valid HF_TOKEN with access "
            "or rerun with --skip_used_corpus_filter."
        ) from exc

    used_ids: set[str] = set()
    for item in ds:
        used_ids.add(str(item["id"]))
        if len(used_ids) % 100_000 == 0:
            print(f"  Loaded {len(used_ids)} used corpus ids...")
    return used_ids


def main(
    input_path: str,
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    msmarco_extra_docs: int,
    nq_doc_id_start: int,
    seed: int,
    skip_used_corpus_filter: bool,
    used_corpus_dataset: str,
    hf_token: Optional[str],
    batch_size: int,
) -> None:
    for path in (queries_path, corpus_path, relevant_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    queries_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    corpus_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    relevant_schema = pa.schema([("query_id", pa.string()), ("document_id", pa.string())])

    queries_writer = corpus_writer = relevant_writer = None

    try:
        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)

        print(f"Processing Natural Questions JSONL: {input_path}")
        nq_stats = write_natural_questions_parquets(
            input_path=input_path,
            queries_writer=queries_writer,
            corpus_writer=corpus_writer,
            relevant_writer=relevant_writer,
            queries_schema=queries_schema,
            corpus_schema=corpus_schema,
            relevant_schema=relevant_schema,
            nq_doc_id_start=nq_doc_id_start,
            batch_size=batch_size,
        )
        print(
            "Finished Natural Questions. "
            f"Rows: {nq_stats.processed_rows}, queries: {nq_stats.written_queries}, "
            f"corpus docs: {nq_stats.written_corpus_docs}, relevant pairs: {nq_stats.written_relevant_pairs}."
        )
        if (
            nq_stats.skipped_bad_json
            or nq_stats.skipped_bad_id
            or nq_stats.skipped_empty_query
            or nq_stats.skipped_bad_texts_or_labels
        ):
            print(
                "Skipped Natural Questions rows. "
                f"bad_json={nq_stats.skipped_bad_json}, bad_id={nq_stats.skipped_bad_id}, "
                f"empty_query={nq_stats.skipped_empty_query}, "
                f"bad_texts_or_labels={nq_stats.skipped_bad_texts_or_labels}."
            )

        used_ids: set[str] = set()
        if skip_used_corpus_filter:
            print("Skipping used-corpus filter for MS MARCO docs.")
        else:
            print(f"Loading used corpus ids from {used_corpus_dataset}...")
            used_ids = load_used_corpus_ids(used_corpus_dataset, hf_token)
            print(f"Loaded {len(used_ids)} used corpus ids to exclude.")

        print("Downloading clarin-knext/msmarco-pl corpus.jsonl.gz...")
        msmarco_corpus_path = download_msmarco_pl_corpus(hf_token)

        print(f"Sampling {msmarco_extra_docs} extra MS MARCO docs...")
        reservoir, eligible_seen, skipped_used = reservoir_sample_msmarco_docs(
            docs=iter_jsonl_gz(msmarco_corpus_path),
            sample_size=msmarco_extra_docs,
            used_ids=used_ids,
            rng=random.Random(seed),
        )

        if len(reservoir) < msmarco_extra_docs:
            print(
                f"Warning: requested {msmarco_extra_docs} MS MARCO docs, "
                f"but only {len(reservoir)} eligible docs were available."
            )

        written_extra = 0
        start = 0
        while start < len(reservoir):
            chunk = reservoir[start : start + batch_size]
            corpus_writer.write_table(pa.Table.from_pylist(chunk, schema=corpus_schema))
            written_extra += len(chunk)
            start += batch_size
            if written_extra % (batch_size * 10) == 0:
                print(f"  Written {written_extra} sampled MS MARCO docs...")

        print(
            "Finished MS MARCO sampling. "
            f"Eligible seen: {eligible_seen}, skipped used: {skipped_used}, written extra: {written_extra}."
        )
    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    print("All Parquet files are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert translated Natural Questions plus sampled Polish MS MARCO docs to Parquet."
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"))
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"))
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH"))
    parser.add_argument("--msmarco_extra_docs", type=int, default=1_500_000)
    parser.add_argument("--nq_doc_id_start", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_used_corpus_filter", action="store_true")
    parser.add_argument("--used_corpus_dataset", type=str, default="minehard/negatives2")
    parser.add_argument("--hf_token", type=str, default=config("HF_TOKEN", default=None))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        msmarco_extra_docs=args.msmarco_extra_docs,
        nq_doc_id_start=args.nq_doc_id_start,
        seed=args.seed,
        skip_used_corpus_filter=args.skip_used_corpus_filter,
        used_corpus_dataset=args.used_corpus_dataset,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
    )
