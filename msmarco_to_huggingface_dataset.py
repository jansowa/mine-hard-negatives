import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from decouple import config


BATCH_SIZE = 10_000  # Process N rows at a time for better I/O performance


def main(
        queries_path: str,
        corpus_path: str,
        relevant_path: str
) -> None:
    # Ensure output directories exist
    for path in [queries_path, corpus_path, relevant_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define schemas for Parquet files as requested
    queries_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    corpus_schema: pa.Schema = pa.schema([("id", pa.int32()), ("text", pa.string())])
    relevant_schema: pa.Schema = pa.schema([("query_id", pa.int32()), ("document_id", pa.int32())])

    queries_writer = None
    corpus_writer = None
    relevant_writer = None

    query_id_map = {}  # To map original string query IDs to new int32 IDs

    try:
        # --- Process Queries ---
        print("Processing queries from BeIR/msmarco (queries subset)...")
        # The 'queries' subset in BeIR/msmarco usually has a 'train' split (or equivalent default)
        # For BeIR/msmarco, 'queries' and 'corpus' subsets have data in their 'train' split.
        hf_queries = load_dataset("BeIR/msmarco", "queries", split="queries", trust_remote_code=True)
        total_queries = len(hf_queries)

        queries_writer = pq.ParquetWriter(queries_path, queries_schema)
        current_new_query_id = 0
        batch = []

        for item in hf_queries:
            original_query_id_str = item["_id"]  # This is a string, e.g., "0", "1", "12345"
            text = item["text"]

            query_id_map[original_query_id_str] = current_new_query_id
            batch.append({"id": current_new_query_id, "text": text})

            current_new_query_id += 1

            if len(batch) >= BATCH_SIZE:
                table = pa.Table.from_pylist(batch, schema=queries_schema)
                queries_writer.write_table(table)
                batch = []
                print(f"  Written {current_new_query_id}/{total_queries} queries...")

        if batch:  # Write any remaining rows
            table = pa.Table.from_pylist(batch, schema=queries_schema)
            queries_writer.write_table(table)
        print(f"Finished processing queries. Total queries: {current_new_query_id}. Saved to {queries_path}")

        # --- Process Corpus ---
        print("Processing corpus from BeIR/msmarco (corpus subset)...")
        hf_corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
        total_corpus = len(hf_corpus)

        corpus_writer = pq.ParquetWriter(corpus_path, corpus_schema)
        current_new_corpus_id = 0
        batch = []

        # The new corpus ID will be its 0-indexed position in this dataset.
        # This is crucial because msmarco-qrels corpus_id typically refers to this 0-indexed position.
        for item in hf_corpus:
            text = item["text"]  # We only need the text
            # The original corpus "_id" (e.g., "msmarco_doc_0_0") is not directly used for qrels mapping
            # with this strategy; the new ID is its sequential position.

            batch.append({"id": current_new_corpus_id, "text": text})
            current_new_corpus_id += 1

            if len(batch) >= BATCH_SIZE:
                table = pa.Table.from_pylist(batch, schema=corpus_schema)
                corpus_writer.write_table(table)
                batch = []
                print(f"  Written {current_new_corpus_id}/{total_corpus} corpus documents...")

        if batch:  # Write any remaining rows
            table = pa.Table.from_pylist(batch, schema=corpus_schema)
            corpus_writer.write_table(table)
        print(f"Finished processing corpus. Total documents: {current_new_corpus_id}. Saved to {corpus_path}")

        # --- Process Relevant Qrels ---
        print("Processing qrels from BeIR/msmarco-qrels...")
        hf_qrels = load_dataset("BeIR/msmarco-qrels", split="train", trust_remote_code=True)
        total_qrels = len(hf_qrels)

        relevant_writer = pq.ParquetWriter(relevant_path, relevant_schema)
        batch = []
        processed_qrels_count = 0
        skipped_qrels_count = 0

        for item in hf_qrels:
            # 'query-id' from qrels is int32 (e.g., 1100500). We need its string form for map lookup.
            original_qrels_query_id_str = str(item["query-id"])
            # 'corpus-id' from qrels is int32 (e.g., 2211018). This is treated as the 0-indexed ID
            # for the documents in corpus.parquet we just created.
            original_qrels_corpus_id = item["corpus-id"]

            new_query_id = query_id_map.get(original_qrels_query_id_str)
            new_document_id = original_qrels_corpus_id  # This is already the 0-indexed new ID

            if new_query_id is not None:
                # We also check if the document_id is within the bounds of our processed corpus
                if new_document_id < current_new_corpus_id:  # type: ignore
                    batch.append({"query_id": new_query_id, "document_id": new_document_id})
                    processed_qrels_count += 1
                else:
                    # This case should ideally not happen if qrels are consistent with the corpus size
                    print(
                        f"  Warning: Corpus ID '{new_document_id}' from qrels is out of bounds for processed corpus (size: {current_new_corpus_id}). Skipping.")  # type: ignore
                    skipped_qrels_count += 1
            else:
                # print(f"  Warning: Query ID '{original_qrels_query_id_str}' from qrels not found in processed queries. Skipping.")
                skipped_qrels_count += 1

            if len(batch) >= BATCH_SIZE:
                table = pa.Table.from_pylist(batch, schema=relevant_schema)
                relevant_writer.write_table(table)
                batch = []
                print(f"  Written {processed_qrels_count}/{total_qrels} relevant pairs...")

        if batch:  # Write any remaining rows
            table = pa.Table.from_pylist(batch, schema=relevant_schema)
            relevant_writer.write_table(table)

        if skipped_qrels_count > 0:
            print(
                f"  Note: Skipped {skipped_qrels_count} qrels entries due to missing query IDs or out-of-bounds corpus IDs.")
        print(
            f"Finished processing relevant pairs. Total pairs written: {processed_qrels_count}. Saved to {relevant_path}")

    finally:
        if queries_writer:
            queries_writer.close()
        if corpus_writer:
            corpus_writer.close()
        if relevant_writer:
            relevant_writer.close()

    print("All datasets have been processed and saved in Parquet format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processing Hugging Face BeIR/msmarco datasets to Parquet (queries, corpus, relevant)")
    # Parameters are configured as per your example, removing input_file_path
    parser.add_argument(
        "--queries_path",
        type=str,
        required=False,
        help="Output path for the queries file (parquet)",
        default=config("QUERIES_PATH")  # Uses decouple or fallback
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=False,
        help="Output path for the corpus file (parquet)",
        default=config("CORPUS_PATH")  # Uses decouple or fallback
    )
    parser.add_argument(
        "--relevant_path",
        type=str,
        required=False,
        help="Output path for the relevant file (parquet)",
        default=config("RELEVANT_PATH")  # Uses decouple or fallback
    )
    args = parser.parse_args()

    main(
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path
    )