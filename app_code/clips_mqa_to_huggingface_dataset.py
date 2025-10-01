import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from decouple import config

BATCH_SIZE = 10_000


def main(
    queries_path: str,
    corpus_path: str,
    relevant_path: str
) -> None:
    # Ensure output directories exist
    for path in [queries_path, corpus_path, relevant_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    queries_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    corpus_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    relevant_schema: pa.Schema = pa.schema([("query_id", pa.int32()), ("document_id", pa.int32())])

    queries_writer = None
    corpus_writer = None
    relevant_writer = None

    try:
        print("Ładowanie datasetu: clips/mqa, subset=pl-faq-question, split=train ...")
        ds = load_dataset("clips/mqa", "pl-faq-question", split="train", trust_remote_code=True)

        total_rows = len(ds)
        print(f"  Wczytano {total_rows} wierszy.")

        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)

        next_query_id = 1
        next_doc_id = 1

        queries_batch = []
        corpus_batch = []
        relevant_batch = []

        kept_questions = 0
        kept_answers = 0
        skipped_questions = 0

        for idx, item in enumerate(ds):
            q_text = (item.get("name") or "").strip()
            answers = item.get("answers") or []

            accepted_answers = []
            for ans in answers:
                try:
                    if bool(ans.get("is_accepted", False)):
                        atext = (ans.get("text") or "").strip()
                        if atext:
                            accepted_answers.append(atext)
                except Exception:
                    continue

            if not q_text or not accepted_answers:
                skipped_questions += 1
                continue

            q_id = next_query_id
            queries_batch.append({"id": q_id, "text": q_text})
            next_query_id += 1
            kept_questions += 1

            for atext in accepted_answers:
                d_id = next_doc_id
                corpus_batch.append({"id": d_id, "text": atext})
                relevant_batch.append({"query_id": q_id, "document_id": d_id})
                next_doc_id += 1
                kept_answers += 1

            if len(queries_batch) >= BATCH_SIZE:
                queries_writer.write_table(pa.Table.from_pylist(queries_batch, schema=queries_schema))
                queries_batch = []
                print(f"  Zapisano {kept_questions} pytań...")

            if len(corpus_batch) >= BATCH_SIZE:
                corpus_writer.write_table(pa.Table.from_pylist(corpus_batch, schema=corpus_schema))
                corpus_batch = []
                print(f"  Zapisano {kept_answers} zaakceptowanych odpowiedzi...")

            if len(relevant_batch) >= BATCH_SIZE:
                relevant_writer.write_table(pa.Table.from_pylist(relevant_batch, schema=relevant_schema))
                relevant_batch = []
                print(f"  Zapisano {kept_answers} powiązań query->answer...")

        if queries_batch:
            queries_writer.write_table(pa.Table.from_pylist(queries_batch, schema=queries_schema))
        if corpus_batch:
            corpus_writer.write_table(pa.Table.from_pylist(corpus_batch, schema=corpus_schema))
        if relevant_batch:
            relevant_writer.write_table(pa.Table.from_pylist(relevant_batch, schema=relevant_schema))

        print("Przetwarzanie zakończone.")
        print(f"  Pytania zachowane: {kept_questions}")
        print(f"  Odpowiedzi zaakceptowane (zapisane): {kept_answers}")
        print(f"  Pytania pominięte (brak zaakceptowanych lub puste): {skipped_questions}")
        print("Zapisano:")
        print(f"  Queries  -> {queries_path}")
        print(f"  Corpus   -> {corpus_path}")
        print(f"  Relevant -> {relevant_path}")

    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    print("Wszystkie pliki Parquet gotowe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Przetwarzanie HF clips/mqa (pl-faq-question/train) do Parquet (queries, corpus, relevant)"
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        required=False,
        help="Ścieżka docelowa dla pliku queries (parquet)",
        default=config("QUERIES_PATH")
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=False,
        help="Ścieżka docelowa dla pliku corpus (parquet)",
        default=config("CORPUS_PATH")
    )
    parser.add_argument(
        "--relevant_path",
        type=str,
        required=False,
        help="Ścieżka docelowa dla pliku relevant (parquet)",
        default=config("RELEVANT_PATH")
    )
    args = parser.parse_args()

    main(
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path
    )