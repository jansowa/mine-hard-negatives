import gzip
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from natural_questions_to_huggingface_dataset import (
    extract_trailing_query_id,
    iter_jsonl_gz,
    reservoir_sample_msmarco_docs,
    write_natural_questions_parquets,
)


def test_extract_trailing_query_id():
    assert extract_trailing_query_id("dataset:train:123") == "123"
    assert extract_trailing_query_id("no-number") is None


def test_write_natural_questions_parquets_uses_string_ids_and_labels(tmp_path):
    rows = [
        {
            "id": "tomaarsen_natural-questions-hard-negatives_triplet-5:train:7",
            "query": "kto wygrał",
            "texts": ["positive", "negative", "also positive"],
            "labels": [1, 0, 1],
        },
        {
            "id": "bad",
            "query": "skip me",
            "texts": ["x"],
            "labels": [1],
        },
        {
            "id": "dataset:train:8",
            "query": "bad labels",
            "texts": ["x", "y"],
            "labels": [1],
        },
    ]

    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"

    queries_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    corpus_schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    relevant_schema = pa.schema([("query_id", pa.string()), ("document_id", pa.string())])

    queries_writer = pq.ParquetWriter(queries_path, queries_schema)
    corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
    relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)
    try:
        stats = write_natural_questions_parquets(
            rows=rows,
            queries_writer=queries_writer,
            corpus_writer=corpus_writer,
            relevant_writer=relevant_writer,
            queries_schema=queries_schema,
            corpus_schema=corpus_schema,
            relevant_schema=relevant_schema,
            nq_doc_id_start=10_000_000,
            batch_size=2,
        )
    finally:
        queries_writer.close()
        corpus_writer.close()
        relevant_writer.close()

    queries = pq.read_table(queries_path).to_pylist()
    corpus = pq.read_table(corpus_path).to_pylist()
    relevant = pq.read_table(relevant_path).to_pylist()

    assert stats.written_queries == 1
    assert stats.written_corpus_docs == 3
    assert stats.written_relevant_pairs == 2
    assert stats.skipped_bad_id == 1
    assert stats.skipped_bad_texts_or_labels == 1

    assert queries == [{"id": "7", "text": "kto wygrał"}]
    assert corpus == [
        {"id": "10000000", "text": "positive"},
        {"id": "10000001", "text": "negative"},
        {"id": "10000002", "text": "also positive"},
    ]
    assert relevant == [
        {"query_id": "7", "document_id": "10000000"},
        {"query_id": "7", "document_id": "10000002"},
    ]

    assert pq.read_schema(queries_path).field("id").type == pa.string()
    assert pq.read_schema(corpus_path).field("id").type == pa.string()
    assert pq.read_schema(relevant_path).field("query_id").type == pa.string()
    assert pq.read_schema(relevant_path).field("document_id").type == pa.string()


def test_reservoir_sample_msmarco_docs_filters_used_ids_and_is_deterministic():
    docs = [
        {"_id": str(i), "text": f"text {i}"}
        for i in range(10)
    ]

    sample_1, eligible_seen_1, skipped_used_1 = reservoir_sample_msmarco_docs(
        docs=docs,
        sample_size=4,
        used_ids={"2", "7"},
        rng=random.Random(123),
    )
    sample_2, eligible_seen_2, skipped_used_2 = reservoir_sample_msmarco_docs(
        docs=docs,
        sample_size=4,
        used_ids={"2", "7"},
        rng=random.Random(123),
    )

    assert sample_1 == sample_2
    assert eligible_seen_1 == eligible_seen_2 == 8
    assert skipped_used_1 == skipped_used_2 == 2
    assert len(sample_1) == 4
    assert {row["id"] for row in sample_1}.isdisjoint({"2", "7"})


def test_iter_jsonl_gz_reads_records(tmp_path):
    path = tmp_path / "corpus.jsonl.gz"
    rows = [
        {"_id": "1", "text": "one"},
        {"_id": "2", "text": "two"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")

    assert list(iter_jsonl_gz(str(path))) == rows
