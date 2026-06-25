import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from nfcorpus_pl_to_huggingface_dataset import (
    corpus_text_from_row,
    parse_qrels_splits,
    write_nfcorpus_parquets,
)


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_corpus_text_from_row_modes():
    row = {"title": "Tytuł", "text": "Treść"}

    assert corpus_text_from_row(row, "title_text") == "Tytuł\nTreść"
    assert corpus_text_from_row(row, "title") == "Tytuł"
    assert corpus_text_from_row(row, "text") == "Treść"


def test_parse_qrels_splits_accepts_validation_alias():
    assert parse_qrels_splits("train, validation,dev,test") == ["train", "validation", "dev", "test"]


def test_write_nfcorpus_parquets_maps_and_filters_rows(tmp_path):
    queries_source = tmp_path / "queries.jsonl.gz"
    corpus_source = tmp_path / "corpus.jsonl.gz"
    train_qrels = tmp_path / "train.tsv"
    dev_qrels = tmp_path / "dev.tsv"

    write_jsonl_gz(
        queries_source,
        [
            {"_id": "PLAIN-1", "text": "pierwsze pytanie"},
            {"_id": "PLAIN-2", "text": "drugie pytanie"},
            {"_id": "EMPTY", "text": ""},
        ],
    )
    write_jsonl_gz(
        corpus_source,
        [
            {"_id": "MED-1", "title": "Tytuł 1", "text": "Treść 1"},
            {"_id": "MED-2", "title": "Tytuł 2", "text": "Treść 2"},
            {"_id": "EMPTY", "title": "", "text": ""},
        ],
    )
    write_tsv(
        train_qrels,
        [
            "query-id\tcorpus-id\tscore",
            "PLAIN-1\tMED-1\t1",
            "PLAIN-1\tMED-1\t1",
            "PLAIN-1\tMED-2\t0",
            "MISSING\tMED-2\t1",
        ],
    )
    write_tsv(
        dev_qrels,
        [
            "query-id\tcorpus-id\tscore",
            "PLAIN-2\tMED-2\t2",
            "PLAIN-2\tMISSING\t1",
        ],
    )

    queries_path = tmp_path / "queries.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    relevant_path = tmp_path / "relevant.parquet"

    stats = write_nfcorpus_parquets(
        queries_jsonl_gz_path=str(queries_source),
        corpus_jsonl_gz_path=str(corpus_source),
        qrels_by_split={"train": str(train_qrels), "validation": str(dev_qrels)},
        queries_path=str(queries_path),
        corpus_path=str(corpus_path),
        relevant_path=str(relevant_path),
        corpus_text_mode="title_text",
        min_qrel_score=1,
        batch_size=2,
    )

    assert pq.read_table(queries_path).to_pylist() == [
        {"id": "PLAIN-1", "text": "pierwsze pytanie"},
        {"id": "PLAIN-2", "text": "drugie pytanie"},
    ]
    assert pq.read_table(corpus_path).to_pylist() == [
        {"id": "MED-1", "text": "Tytuł 1\nTreść 1"},
        {"id": "MED-2", "text": "Tytuł 2\nTreść 2"},
    ]
    assert pq.read_table(relevant_path).to_pylist() == [
        {"query_id": "PLAIN-1", "document_id": "MED-1", "qrel_score": 1, "qrel_split": "train"},
        {"query_id": "PLAIN-2", "document_id": "MED-2", "qrel_score": 2, "qrel_split": "validation"},
    ]

    assert stats.written_queries == 2
    assert stats.written_corpus_docs == 2
    assert stats.written_relevant_pairs == 2
    assert stats.skipped_empty_queries == 1
    assert stats.skipped_empty_corpus_docs == 1
    assert stats.skipped_duplicate_qrels == 1
    assert stats.skipped_qrels_below_min_score == 1
    assert stats.skipped_qrels_missing_query == 1
    assert stats.skipped_qrels_missing_document == 1

    assert pq.read_schema(queries_path).field("id").type == pa.string()
    assert pq.read_schema(corpus_path).field("id").type == pa.string()
    assert pq.read_schema(relevant_path).field("query_id").type == pa.string()
    assert pq.read_schema(relevant_path).field("qrel_score").type == pa.int64()
