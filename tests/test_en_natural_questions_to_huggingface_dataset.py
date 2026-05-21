import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from en_natural_questions_to_huggingface_dataset import (
    first_non_empty_text,
    natural_questions_id_from_position,
    natural_questions_position_from_id,
    normalize_hf_token,
    reservoir_sample_msmarco_docs,
    write_natural_questions_parquets,
)


def test_natural_questions_position_id_is_offset_and_reversible():
    assert natural_questions_id_from_position(0, 10_000_000) == "10000000"
    assert natural_questions_id_from_position(42, 10_000_000) == "10000042"
    assert natural_questions_position_from_id("10000042", 10_000_000) == 42
    assert natural_questions_position_from_id("9999999", 10_000_000) is None
    assert natural_questions_position_from_id("bad", 10_000_000) is None


def test_natural_questions_field_helpers_accept_hf_pair_schema():
    assert first_non_empty_text({"query": "who wrote dune", "question": "ignored"}, ("query", "question")) == (
        "who wrote dune"
    )
    assert first_non_empty_text({"question": "fallback"}, ("query", "question")) == "fallback"
    assert first_non_empty_text({"query": " "}, ("query", "question")) == ""
    assert normalize_hf_token("hf_TOKEN") is None
    assert normalize_hf_token("  hf_actual  ") == "hf_actual"


def test_write_natural_questions_parquets_uses_offset_positions(tmp_path):
    rows = [
        {"query": "who wrote dune", "answer": "Frank Herbert"},
        {"question": " ", "answer": "skip empty question"},
        {"question": "where is oslo", "answer": ""},
        {"query": "capital of france", "answer": "Paris"},
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
            id_offset=10_000_000,
            batch_size=2,
        )
    finally:
        queries_writer.close()
        corpus_writer.close()
        relevant_writer.close()

    queries = pq.read_table(queries_path).to_pylist()
    corpus = pq.read_table(corpus_path).to_pylist()
    relevant = pq.read_table(relevant_path).to_pylist()

    assert stats.processed_rows == 4
    assert stats.written_queries == 2
    assert stats.written_corpus_docs == 2
    assert stats.written_relevant_pairs == 2
    assert stats.skipped_empty_question == 1
    assert stats.skipped_empty_answer == 1

    assert queries == [
        {"id": "10000000", "text": "who wrote dune"},
        {"id": "10000003", "text": "capital of france"},
    ]
    assert corpus == [
        {"id": "10000000", "text": "Frank Herbert"},
        {"id": "10000003", "text": "Paris"},
    ]
    assert relevant == [
        {"query_id": "10000000", "document_id": "10000000"},
        {"query_id": "10000003", "document_id": "10000003"},
    ]


def test_reservoir_sample_msmarco_docs_filters_used_ids_and_empty_rows():
    docs = [
        {"_id": "1", "text": "keep 1"},
        {"_id": "2", "text": "used"},
        {"_id": "", "text": "bad id"},
        {"_id": "3", "text": ""},
        {"_id": "4", "text": "keep 4"},
        {"_id": "5", "text": "keep 5"},
    ]

    sample_1, stats_1 = reservoir_sample_msmarco_docs(
        docs=docs,
        sample_size=2,
        used_ids={"2"},
        rng=random.Random(123),
    )
    sample_2, stats_2 = reservoir_sample_msmarco_docs(
        docs=docs,
        sample_size=2,
        used_ids={"2"},
        rng=random.Random(123),
    )

    assert sample_1 == sample_2
    assert stats_1 == stats_2
    assert stats_1.eligible_seen == 3
    assert stats_1.skipped_used == 1
    assert stats_1.skipped_empty_id == 1
    assert stats_1.skipped_empty_text == 1
    assert len(sample_1) == 2
    assert {row["id"] for row in sample_1}.isdisjoint({"2"})
