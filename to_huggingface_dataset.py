import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ścieżki do plików wejściowego i wyjściowych
input_file_path = 'data/sample_input.jsonl'
queries_path = 'data/queries.parquet'
corpus_path = 'data/corpus.parquet'
relevant_path = 'data/relevant.parquet'

query_id = 0
corpus_id = 0

# Schematy dla Parquet
queries_schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
corpus_schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
relevant_schema = pa.schema([("query_id", pa.int32()), ("document_id", pa.int32())])

# Inicjalizacja pisarzy Parquet
queries_writer = None
corpus_writer = None
relevant_writer = None

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        messages = data.get("messages", [])

        for i in range(0, len(messages) - 1):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]

            if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                # Dane pojedynczego wiersza
                query_row = {"id": query_id, "text": user_msg["content"]}
                corpus_row = {"id": corpus_id, "text": assistant_msg["content"]}
                relevant_row = {"query_id": query_id, "document_id": corpus_id}

                # Konwersja do tabeli Arrow
                query_table = pa.Table.from_pylist([query_row], schema=queries_schema)
                corpus_table = pa.Table.from_pylist([corpus_row], schema=corpus_schema)
                relevant_table = pa.Table.from_pylist([relevant_row], schema=relevant_schema)

                # Inicjalizacja pisarzy jeśli trzeba
                if queries_writer is None:
                    queries_writer = pq.ParquetWriter(queries_path, queries_schema)
                if corpus_writer is None:
                    corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
                if relevant_writer is None:
                    relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)

                # Zapis do plików
                queries_writer.write_table(query_table)
                corpus_writer.write_table(corpus_table)
                relevant_writer.write_table(relevant_table)

                # Inkrementacja ID
                query_id += 1
                corpus_id += 1

# Zamknięcie writerów
if queries_writer:
    queries_writer.close()
if corpus_writer:
    corpus_writer.close()
if relevant_writer:
    relevant_writer.close()

print("Zbiory zostały zapisane w formacie Parquet (strumieniowo).")
