import argparse
import os
import random
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from decouple import config

BATCH_SIZE = 10_000


def write_batch(writer: pq.ParquetWriter, batch: list[dict], schema: pa.Schema) -> None:
    if batch:
        table = pa.Table.from_pylist(batch, schema=schema)
        writer.write_table(table)
        batch.clear()


def main(
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    corpus_max_docs: Optional[int],
    seed: int,
) -> None:
    random.seed(seed)

    for path in (queries_path, corpus_path, relevant_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    queries_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    corpus_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    relevant_schema = pa.schema([("query_id", pa.string()), ("document_id", pa.string())])

    queries_writer = corpus_writer = relevant_writer = None

    try:
        # Write relevant pairs first and collect required document ids for corpus.
        print("Processing qrels...")
        hf_qrels = load_dataset("BeIR/msmarco-qrels", split="train", trust_remote_code=True)

        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)
        rel_batch: list[dict] = []
        processed_qrels_count = 0
        required_doc_ids: set[str] = set()

        for item in hf_qrels:
            qid = str(item["query-id"])
            did = str(item["corpus-id"])
            rel_batch.append({"query_id": qid, "document_id": did})
            required_doc_ids.add(did)
            processed_qrels_count += 1
            if len(rel_batch) >= BATCH_SIZE:
                write_batch(relevant_writer, rel_batch, relevant_schema)
                print(f"  Written {processed_qrels_count} relevant pairs...")

        write_batch(relevant_writer, rel_batch, relevant_schema)
        print(f"Finished qrels. Total pairs: {processed_qrels_count}. Saved to {relevant_path}")

        # Write queries (full set).
        print("Processing queries...")
        hf_queries = load_dataset("BeIR/msmarco", "queries", split="queries", trust_remote_code=True)

        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
        q_batch: list[dict] = []
        processed_queries = 0

        for item in hf_queries:
            qid = str(item["_id"])
            text = item["text"]
            q_batch.append({"id": qid, "text": text})
            processed_queries += 1
            if len(q_batch) >= BATCH_SIZE:
                write_batch(queries_writer, q_batch, queries_schema)
                if processed_queries % (BATCH_SIZE * 10) == 0:
                    print(f"  Written {processed_queries} queries...")

        write_batch(queries_writer, q_batch, queries_schema)
        print(f"Finished queries. Total: {processed_queries}. Saved to {queries_path}")

        # Write corpus with required docs + sampled extras up to limit.
        print("Processing corpus...")
        hf_corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)

        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
        c_batch: list[dict] = []
        written_doc_ids: set[str] = set()

        if corpus_max_docs is None or corpus_max_docs <= 0:
            total_written = 0
            for item in hf_corpus:
                did = str(item["_id"])
                text = item["text"]
                c_batch.append({"id": did, "text": text})
                total_written += 1
                if len(c_batch) >= BATCH_SIZE:
                    write_batch(corpus_writer, c_batch, corpus_schema)
                    if total_written % (BATCH_SIZE * 10) == 0:
                        print(f"  Written {total_written} corpus docs...")
            write_batch(corpus_writer, c_batch, corpus_schema)
            print(f"Finished corpus. Total docs: {total_written}. Saved to {corpus_path}")
        else:
            effective_limit = max(corpus_max_docs, len(required_doc_ids))
            if effective_limit > corpus_max_docs:
                print(
                    f"Requested corpus_max_docs={corpus_max_docs}, "
                    f"but required {len(required_doc_ids)} docs from qrels; using {effective_limit}."
                )

            reservoir_size = max(0, effective_limit - len(required_doc_ids))
            reservoir: list[dict] = []
            seen_optional = 0
            written_required = 0

            for item in hf_corpus:
                did = str(item["_id"])
                text = item["text"]

                if did in required_doc_ids and did not in written_doc_ids:
                    c_batch.append({"id": did, "text": text})
                    written_doc_ids.add(did)
                    written_required += 1
                    if len(c_batch) >= BATCH_SIZE:
                        write_batch(corpus_writer, c_batch, corpus_schema)
                        if written_required % (BATCH_SIZE * 10) == 0:
                            print(f"  Written required {written_required} docs...")
                elif did not in written_doc_ids:
                    if reservoir_size > 0:
                        seen_optional += 1
                        if len(reservoir) < reservoir_size:
                            reservoir.append({"id": did, "text": text})
                        else:
                            # Reservoir sampling: replace with probability reservoir_size/seen_optional.
                            j = random.randint(1, seen_optional)
                            if j <= reservoir_size:
                                replace_idx = random.randint(0, reservoir_size - 1)
                                reservoir[replace_idx] = {"id": did, "text": text}

            write_batch(corpus_writer, c_batch, corpus_schema)

            extra_written = 0
            if reservoir:
                start = 0
                while start < len(reservoir):
                    chunk = reservoir[start : start + BATCH_SIZE]
                    table = pa.Table.from_pylist(chunk, schema=corpus_schema)
                    corpus_writer.write_table(table)
                    extra_written += len(chunk)
                    start += BATCH_SIZE

            total_written = written_required + extra_written
            print(
                f"Finished corpus with limit. Required: {written_required}, "
                f"sampled: {extra_written}, total: {total_written} (limit={effective_limit}). "
                f"Saved to {corpus_path}"
            )

    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    print("All datasets processed and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BeIR/msmarco to Parquet (queries, corpus with limit, relevant)."
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        required=False,
        help="Output path for queries parquet.",
        default=config("QUERIES_PATH"),
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=False,
        help="Output path for corpus parquet.",
        default=config("CORPUS_PATH"),
    )
    parser.add_argument(
        "--relevant_path",
        type=str,
        required=False,
        help="Output path for relevant parquet.",
        default=config("RELEVANT_PATH"),
    )
    parser.add_argument(
        "--corpus_max_docs",
        type=int,
        required=False,
        help="Max docs in corpus parquet. All docs referenced by qrels are always included. "
        "If <=0 or unset, writes full corpus.",
        default=1_500_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed for sampling.",
        default=42,
    )
    args = parser.parse_args()

    main(
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        corpus_max_docs=args.corpus_max_docs,
        seed=args.seed,
    )
