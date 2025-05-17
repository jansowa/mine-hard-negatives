import json
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from decouple import config

def main(
    input_file_path: str,
    queries_path: str,
    corpus_path: str,
    relevant_path: str
) -> None:
    query_id: int = 0
    corpus_id: int = 0

    queries_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    corpus_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    relevant_schema: pa.Schema = pa.schema([("query_id", pa.int32()), ("document_id", pa.int32())])

    queries_writer = None
    corpus_writer = None
    relevant_writer = None

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data: dict = json.loads(line)
            messages: list = data.get("messages", [])

            for i in range(0, len(messages) - 1):
                user_msg: dict = messages[i]
                assistant_msg: dict = messages[i + 1]

                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    query_row: dict = {"id": query_id, "text": user_msg["content"]}
                    corpus_row: dict = {"id": corpus_id, "text": assistant_msg["content"]}
                    relevant_row: dict = {"query_id": query_id, "document_id": corpus_id}

                    query_table: pa.Table = pa.Table.from_pylist([query_row], schema=queries_schema)
                    corpus_table: pa.Table = pa.Table.from_pylist([corpus_row], schema=corpus_schema)
                    relevant_table: pa.Table = pa.Table.from_pylist([relevant_row], schema=relevant_schema)

                    if queries_writer is None:
                        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
                    if corpus_writer is None:
                        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
                    if relevant_writer is None:
                        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)

                    queries_writer.write_table(query_table)
                    corpus_writer.write_table(corpus_table)
                    relevant_writer.write_table(relevant_table)

                    query_id += 1
                    corpus_id += 1

    if queries_writer:
        queries_writer.close()
    if corpus_writer:
        corpus_writer.close()
    if relevant_writer:
        relevant_writer.close()

    print("Zbiory zostały zapisane w formacie Parquet (strumieniowo).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przetwarzanie pliku wejściowego na Parquet (queries, corpus, relevant)")
    parser.add_argument("--input_file_path", type=str, required=False, help="Ścieżka do pliku wejściowego (jsonl)", default=config("INPUT_FILE_PATH"))
    parser.add_argument("--queries_path", type=str, required=False, help="Ścieżka wyjściowa do pliku queries (parquet)", default=config("QUERIES_PATH"))
    parser.add_argument("--corpus_path", type=str, required=False, help="Ścieżka wyjściowa do pliku corpus (parquet)", default=config("CORPUS_PATH"))
    parser.add_argument("--relevant_path", type=str, required=False, help="Ścieżka wyjściowa do pliku relevant (parquet)", default=config("RELEVANT_PATH"))
    args = parser.parse_args()
    main(
        input_file_path=args.input_file_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path
    )