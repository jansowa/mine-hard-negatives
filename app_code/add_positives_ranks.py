import os
import argparse
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from models import get_reranker_model, rerank
from decouple import config

def process_relevant(
    queries_path: str,
    corpus_path: str,
    relevant_path: str,
    output_path: str,
    chunk_size: int,
    reranker_batch_size: int,
    reranker_model_name: str
) -> None:
    output_dir: str = os.path.dirname(output_path) if output_path else "."

    print("Starting creation of relevant_with_score.parquet file...")

    try:
        print(f"Loading data from:\n - {queries_path} (columns: id, text)\n - {corpus_path} (columns: id, text)")
        queries_df: pd.DataFrame = pd.read_parquet(queries_path, columns=['id', 'text'])
        queries_df = queries_df.rename(columns={'text': 'query_text'}).set_index('id')

        corpus_df: pd.DataFrame = pd.read_parquet(corpus_path, columns=['id', 'text'])
        corpus_df = corpus_df.rename(columns={'text': 'document_text'}).set_index('id')
        print("Files queries.parquet and corpus.parquet loaded and prepared.")
        print(f"Queries DataFrame shape: {queries_df.shape}, Corpus DataFrame shape: {corpus_df.shape}")

        tokenizer, reranker = get_reranker_model(reranker_model_name)
        print(f"Reranker loaded.")

        relevant_extended_schema: pa.Schema = pa.schema([
            ('query_id', pa.int32()),
            ('document_id', pa.int32()),
            ('positive_ranking', pa.float32())
        ])
        print("Schema for the output file defined.")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        writer: Optional[pq.ParquetWriter] = pq.ParquetWriter(output_path, relevant_extended_schema)
        print(f"Started writing to file: {output_path}")

        print(f"Processing file {relevant_path} in chunks of size {chunk_size}...")
        parquet_file_relevant = pq.ParquetFile(relevant_path)
        total_batches: int = parquet_file_relevant.num_row_groups
        print(f"Total number of batches in relevant.parquet: {total_batches}")
        processed_pairs_count: int = 0

        for i, batch in enumerate(
            tqdm(
                parquet_file_relevant.iter_batches(batch_size=chunk_size, columns=['query_id', 'document_id']),
                desc="Processing relevant.parquet chunks"
            )):
            relevant_chunk_df: pd.DataFrame = batch.to_pandas()
            print(f"Processing chunk {i+1} ({len(relevant_chunk_df)} rows)...")

            if relevant_chunk_df.empty:
                continue

            merged_chunk_df: pd.DataFrame = pd.merge(
                relevant_chunk_df, queries_df,
                left_on='query_id', right_index=True, how='inner'
            )
            merged_chunk_df = pd.merge(
                merged_chunk_df, corpus_df,
                left_on='document_id', right_index=True, how='inner'
            )
            if merged_chunk_df.empty:
                continue
            
            scores_chunk = rerank(
                tokenizer, reranker,
                merged_chunk_df['query_text'].values.tolist(),
                merged_chunk_df['document_text'].values.tolist(),
                batch_size=reranker_batch_size
            )
            

            print(f"Chunk {i+1} processed: {len(scores_chunk)} scores generated.")
            result_chunk_df: pd.DataFrame = pd.DataFrame({
                'query_id': merged_chunk_df['query_id'],
                'document_id': merged_chunk_df['document_id'],
                'positive_ranking': scores_chunk
            })
            print(f"Result chunk DataFrame shape: {result_chunk_df.shape}")

            result_table_chunk: pa.Table = pa.Table.from_pandas(result_chunk_df, schema=relevant_extended_schema, preserve_index=False)
            writer.write_table(result_table_chunk)
            processed_pairs_count += len(result_chunk_df)

            del relevant_chunk_df, merged_chunk_df, scores_chunk, result_chunk_df, result_table_chunk

        writer.close()
        print(f"\nSuccessfully created file: {output_path}")
        print(f"Processed and saved a total of {processed_pairs_count} question-document pairs.")

        if processed_pairs_count > 0:
            print("\nPreview of the first 5 rows of relevant with score parquet file:")
            print(pd.read_parquet(output_path).head())
        else:
            print("No pairs were processed, the output file may be empty or contain no data.")

    except FileNotFoundError as e:
        print(f"ERROR: One of the input files was not found: {e}. Make sure the files exist at the specified paths.")
    except KeyError as e:
        print(f"ERROR: Missing expected column in one of the Parquet files: {e}. Check the structure of the input files.")
    except AttributeError as e:
        if 'reranker' in str(e) and hasattr(e, 'obj') and e.obj is None:
            print(f"ERROR: The 'reranker' object was not properly initialized (is None). Details: {e}")
        elif 'reranker' in str(e):
            print(f"ERROR: Problem with the 'reranker' object. Make sure it is properly loaded and has a 'predict' method. Details: {e}")
        else:
            print(f"ERROR: An attribute problem occurred: {e}")
    except Exception as e:
        import traceback
        print(f"ERROR: An unexpected error occurred during processing: {e}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        if 'writer' in locals() and writer is not None and writer.is_open:
            writer.close()
            print("ParquetWriter was closed in the finally block.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring relevant pairs and saving to Parquet.")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"), help="Path to parquet file with queries")
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"), help="Path to parquet file with corpus")
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH"), help="Path to parquet file with relevant connections")
    parser.add_argument("--output_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"), help="Path to output parquet file (relevant with positive scores)")
    parser.add_argument("--chunk_size", type=int, default=config("PROCESSING_CHUNK_SIZE", cast=int), help="Chunk size for processing")
    parser.add_argument("--reranker_batch_size", type=int, default=config("RERANKER_BATCH_SIZE", cast=int), help="Batch size for reranker")
    parser.add_argument("--reranker_model_name", type=str, default=config("RERANKER_NAME"), help="Name of the reranker model")
    args = parser.parse_args()

    process_relevant(
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        relevant_path=args.relevant_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        reranker_batch_size=args.reranker_batch_size,
        reranker_model_name=args.reranker_model_name
    )