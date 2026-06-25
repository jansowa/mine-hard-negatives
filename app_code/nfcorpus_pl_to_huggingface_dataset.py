import argparse
import csv
import gzip
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq
from decouple import config
from huggingface_hub import hf_hub_download

BATCH_SIZE = 10_000
DEFAULT_NFCORPUS_DATASET = "clarin-knext/nfcorpus-pl"
DEFAULT_NFCORPUS_QRELS_DATASET = "clarin-knext/nfcorpus-pl-qrels"
QRELS_SPLIT_TO_FILENAME = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}


@dataclass
class NfCorpusStats:
    processed_queries: int = 0
    written_queries: int = 0
    skipped_empty_queries: int = 0
    processed_corpus_docs: int = 0
    written_corpus_docs: int = 0
    skipped_empty_corpus_docs: int = 0
    processed_qrels: int = 0
    written_relevant_pairs: int = 0
    skipped_qrels_below_min_score: int = 0
    skipped_qrels_missing_query: int = 0
    skipped_qrels_missing_document: int = 0
    skipped_duplicate_qrels: int = 0
    qrels_splits: list[str] = field(default_factory=list)


def write_batch(writer: pq.ParquetWriter, batch: list[dict], schema: pa.Schema) -> None:
    if not batch:
        return
    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    batch.clear()


def parse_qrels_splits(raw_splits: str) -> list[str]:
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("At least one qrels split is required")
    unknown = sorted(set(splits) - set(QRELS_SPLIT_TO_FILENAME))
    if unknown:
        known = ", ".join(sorted(QRELS_SPLIT_TO_FILENAME))
        raise ValueError(f"Unknown nfcorpus qrels split(s): {', '.join(unknown)}. Known splits: {known}")
    return splits


def optional_text(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def iter_jsonl_gz(path: str) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def iter_qrels_tsv(path: str, split: str) -> Iterable[dict]:
    with open(path, encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        required_columns = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: query-id, corpus-id, score")
        for row in reader:
            yield {
                "query_id": str(row["query-id"]),
                "document_id": str(row["corpus-id"]),
                "qrel_score": int(row["score"]),
                "qrel_split": split,
            }


def corpus_text_from_row(row: dict, text_mode: str) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if text_mode == "title_text":
        return "\n".join(part for part in (title, text) if part)
    if text_mode == "title":
        return title
    if text_mode == "text":
        return text
    raise ValueError(f"Unknown corpus text mode: {text_mode!r}")


def write_queries(
    rows: Iterable[dict],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    batch_size: int,
    stats: NfCorpusStats,
) -> set[str]:
    query_ids: set[str] = set()
    batch: list[dict] = []
    for row in rows:
        stats.processed_queries += 1
        query_id = str(row.get("_id") or "")
        text = str(row.get("text") or "").strip()
        if not query_id or not text:
            stats.skipped_empty_queries += 1
            continue
        query_ids.add(query_id)
        batch.append({"id": query_id, "text": text})
        stats.written_queries += 1
        if len(batch) >= batch_size:
            write_batch(writer, batch, schema)
            print(f"  Written {stats.written_queries:,} queries...")
    write_batch(writer, batch, schema)
    return query_ids


def write_corpus(
    rows: Iterable[dict],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    batch_size: int,
    text_mode: str,
    stats: NfCorpusStats,
) -> set[str]:
    document_ids: set[str] = set()
    batch: list[dict] = []
    for row in rows:
        stats.processed_corpus_docs += 1
        document_id = str(row.get("_id") or "")
        text = corpus_text_from_row(row, text_mode)
        if not document_id or not text:
            stats.skipped_empty_corpus_docs += 1
            continue
        document_ids.add(document_id)
        batch.append({"id": document_id, "text": text})
        stats.written_corpus_docs += 1
        if len(batch) >= batch_size:
            write_batch(writer, batch, schema)
            print(f"  Written {stats.written_corpus_docs:,} corpus documents...")
    write_batch(writer, batch, schema)
    return document_ids


def write_relevant(
    qrels_by_split: dict[str, str],
    query_ids: set[str],
    document_ids: set[str],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    batch_size: int,
    min_qrel_score: int,
    stats: NfCorpusStats,
) -> None:
    batch: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for split, path in qrels_by_split.items():
        stats.qrels_splits.append(split)
        for row in iter_qrels_tsv(path, split):
            stats.processed_qrels += 1
            if row["qrel_score"] < min_qrel_score:
                stats.skipped_qrels_below_min_score += 1
                continue
            if row["query_id"] not in query_ids:
                stats.skipped_qrels_missing_query += 1
                continue
            if row["document_id"] not in document_ids:
                stats.skipped_qrels_missing_document += 1
                continue

            pair = (row["query_id"], row["document_id"])
            if pair in seen_pairs:
                stats.skipped_duplicate_qrels += 1
                continue
            seen_pairs.add(pair)

            batch.append(row)
            stats.written_relevant_pairs += 1
            if len(batch) >= batch_size:
                write_batch(writer, batch, schema)
                print(f"  Written {stats.written_relevant_pairs:,} relevant pairs...")
    write_batch(writer, batch, schema)


def download_nfcorpus_files(
    dataset_name: str,
    qrels_dataset_name: str,
    qrels_splits: list[str],
    hf_cache_dir: str | None,
    hf_token: str | None,
) -> tuple[str, str, dict[str, str]]:
    queries_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename="queries.jsonl.gz",
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    corpus_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename="corpus.jsonl.gz",
        cache_dir=hf_cache_dir,
        token=hf_token,
    )
    qrels_paths = {
        split: hf_hub_download(
            repo_id=qrels_dataset_name,
            repo_type="dataset",
            filename=QRELS_SPLIT_TO_FILENAME[split],
            cache_dir=hf_cache_dir,
            token=hf_token,
        )
        for split in qrels_splits
    }
    return queries_path, corpus_path, qrels_paths


def write_nfcorpus_parquets(
    queries_jsonl_gz_path: str,
    corpus_jsonl_gz_path: str,
    qrels_by_split: dict[str, str],
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    corpus_text_mode: str,
    min_qrel_score: int,
    batch_size: int,
) -> NfCorpusStats:
    for output_path in (queries_path, corpus_path, relevant_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    queries_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    corpus_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    relevant_schema = pa.schema(
        [
            ("query_id", pa.string()),
            ("document_id", pa.string()),
            ("qrel_score", pa.int64()),
            ("qrel_split", pa.string()),
        ]
    )
    stats = NfCorpusStats()
    queries_writer = corpus_writer = relevant_writer = None

    try:
        print("Processing nfcorpus-pl queries...")
        queries_writer = pq.ParquetWriter(queries_path, queries_schema, compression="zstd", use_dictionary=True)
        query_ids = write_queries(
            iter_jsonl_gz(queries_jsonl_gz_path),
            queries_writer,
            queries_schema,
            batch_size,
            stats,
        )

        print("Processing nfcorpus-pl corpus...")
        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema, compression="zstd", use_dictionary=True)
        document_ids = write_corpus(
            iter_jsonl_gz(corpus_jsonl_gz_path),
            corpus_writer,
            corpus_schema,
            batch_size,
            corpus_text_mode,
            stats,
        )

        print("Processing nfcorpus-pl qrels...")
        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema, compression="zstd", use_dictionary=True)
        write_relevant(
            qrels_by_split=qrels_by_split,
            query_ids=query_ids,
            document_ids=document_ids,
            writer=relevant_writer,
            schema=relevant_schema,
            batch_size=batch_size,
            min_qrel_score=min_qrel_score,
            stats=stats,
        )
    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    return stats


def main(
    dataset_name: str,
    qrels_dataset_name: str,
    qrels_splits: list[str],
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    corpus_text_mode: str,
    min_qrel_score: int,
    hf_cache_dir: str | None,
    hf_token: str | None,
    batch_size: int,
) -> None:
    queries_jsonl_gz_path, corpus_jsonl_gz_path, qrels_paths = download_nfcorpus_files(
        dataset_name=dataset_name,
        qrels_dataset_name=qrels_dataset_name,
        qrels_splits=qrels_splits,
        hf_cache_dir=hf_cache_dir,
        hf_token=hf_token,
    )
    stats = write_nfcorpus_parquets(
        queries_jsonl_gz_path=queries_jsonl_gz_path,
        corpus_jsonl_gz_path=corpus_jsonl_gz_path,
        qrels_by_split=qrels_paths,
        queries_path=queries_path,
        corpus_path=corpus_path,
        relevant_path=relevant_path,
        corpus_text_mode=corpus_text_mode,
        min_qrel_score=min_qrel_score,
        batch_size=batch_size,
    )

    print("nfcorpus-pl Parquet artifacts ready.")
    print(f"  Queries:  {stats.written_queries:,} -> {queries_path}")
    print(f"  Corpus:   {stats.written_corpus_docs:,} -> {corpus_path}")
    print(f"  Relevant: {stats.written_relevant_pairs:,} -> {relevant_path}")
    if stats.skipped_qrels_missing_query or stats.skipped_qrels_missing_document:
        print(
            "  Skipped qrels with missing ids: "
            f"queries={stats.skipped_qrels_missing_query:,}, "
            f"documents={stats.skipped_qrels_missing_document:,}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert clarin-knext/nfcorpus-pl + nfcorpus-pl-qrels to pipeline Parquet artifacts."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=config("NFCORPUS_DATASET", default=DEFAULT_NFCORPUS_DATASET),
    )
    parser.add_argument(
        "--qrels_dataset_name",
        type=str,
        default=config("NFCORPUS_QRELS_DATASET", default=DEFAULT_NFCORPUS_QRELS_DATASET),
    )
    parser.add_argument(
        "--qrels_splits",
        type=str,
        default=config("NFCORPUS_QRELS_SPLITS", default="train,validation,test"),
        help="Comma-separated qrels splits to merge. Use train, validation/dev, and/or test.",
    )
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH", default="data/queries.parquet"))
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH", default="data/corpus.parquet"))
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH", default="data/relevant.parquet"))
    parser.add_argument(
        "--corpus_text_mode",
        type=str,
        choices=("title_text", "text", "title"),
        default=config("NFCORPUS_CORPUS_TEXT_MODE", default="title_text"),
        help="How to map nfcorpus document title/text into the pipeline corpus text column.",
    )
    parser.add_argument(
        "--min_qrel_score",
        type=int,
        default=config("NFCORPUS_MIN_QREL_SCORE", cast=int, default=1),
        help="Keep qrels with score >= this value.",
    )
    parser.add_argument("--hf_cache_dir", type=str, default=config("NFCORPUS_HF_CACHE_DIR", default=None))
    parser.add_argument("--hf_token", type=str, default=config("HF_TOKEN", default=None))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config("NFCORPUS_BATCH_SIZE", cast=int, default=BATCH_SIZE),
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        qrels_dataset_name=args.qrels_dataset_name,
        qrels_splits=parse_qrels_splits(args.qrels_splits),
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        corpus_text_mode=args.corpus_text_mode,
        min_qrel_score=args.min_qrel_score,
        hf_cache_dir=optional_text(args.hf_cache_dir),
        hf_token=optional_text(args.hf_token),
        batch_size=args.batch_size,
    )
