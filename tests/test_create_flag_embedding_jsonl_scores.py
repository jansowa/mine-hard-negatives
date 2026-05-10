import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from create_flag_embedding_jsonl import process_negatives_streaming


def test_process_negatives_streaming_reads_custom_negative_score_column(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    queries_path = tmp_path / "queries.parquet"
    relevant_path = tmp_path / "relevant.parquet"
    negatives_path = tmp_path / "negatives.parquet"
    output_path = tmp_path / "train.jsonl"

    pd.DataFrame(
        {
            "id": ["p1", "n1"],
            "text": ["positive text", "negative text"],
        }
    ).to_parquet(corpus_path, index=False)
    pd.DataFrame({"id": ["q1"], "text": ["question"]}).to_parquet(queries_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["p1"],
            "positive_ranking": [1.0],
        }
    ).to_parquet(relevant_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"],
            "document_id": ["n1"],
            "final_ranking": [0.5],
        }
    ).to_parquet(negatives_path, index=False)

    process_negatives_streaming(
        corpus_path=str(corpus_path),
        queries_path=str(queries_path),
        relevant_path=str(relevant_path),
        negatives_path=str(negatives_path),
        output_path=str(output_path),
        num_negatives=1,
        positive_score_column="positive_ranking",
        negative_score_column="final_ranking",
        beta=0.5,
        u_floor=0.0,
        max_neg_reuse=10,
        corpus_sqlite_path=str(tmp_path / "corpus.sqlite"),
        negcount_sqlite_path=str(tmp_path / "negcount.sqlite"),
        query_chunk_size=1,
        oversample_factor=1,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["neg_id"] == ["n1"]
    assert rows[0]["neg_scores"] == [0.5]
