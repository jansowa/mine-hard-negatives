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

    print("Rozpoczynam tworzenie pliku relevant_with_score.parquet...")

    try:
        print(f"Wczytywanie danych z:\n - {queries_path} (kolumny: id, text)\n - {corpus_path} (kolumny: id, text)")
        queries_df: pd.DataFrame = pd.read_parquet(queries_path, columns=['id', 'text'])
        queries_df = queries_df.rename(columns={'text': 'query_text'}).set_index('id')

        corpus_df: pd.DataFrame = pd.read_parquet(corpus_path, columns=['id', 'text'])
        corpus_df = corpus_df.rename(columns={'text': 'document_text'}).set_index('id')
        print("Pliki queries.parquet i corpus.parquet wczytane i przygotowane.")

        tokenizer, reranker = get_reranker_model(reranker_model_name)
        print(f"Reranker załadowany. Urządzenie: {reranker.device}")

        relevant_extended_schema: pa.Schema = pa.schema([
            ('query_id', pa.int32()),
            ('document_id', pa.int32()),
            ('positive_ranking', pa.float32())
        ])
        print("Zdefiniowano schemat dla pliku wyjściowego.")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Utworzono katalog docelowy: {output_dir}")

        writer: Optional[pq.ParquetWriter] = pq.ParquetWriter(output_path, relevant_extended_schema)
        print(f"Rozpoczęto zapis do pliku: {output_path}")

        print(f"Przetwarzanie pliku {relevant_path} w chunkach o rozmiarze {chunk_size}...")
        parquet_file_relevant = pq.ParquetFile(relevant_path)
        total_batches: int = parquet_file_relevant.num_row_groups

        processed_pairs_count: int = 0

        for i, batch in enumerate(
            tqdm(
                parquet_file_relevant.iter_batches(batch_size=chunk_size, columns=['query_id', 'document_id']),
                desc="Przetwarzanie chunków relevant.parquet"
            )):
            relevant_chunk_df: pd.DataFrame = batch.to_pandas()
            print(f"Przetwarzanie chunka {i+1} ({len(relevant_chunk_df)} wierszy)...")

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

            result_chunk_df: pd.DataFrame = pd.DataFrame({
                'query_id': merged_chunk_df['query_id'],
                'document_id': merged_chunk_df['document_id'],
                'positive_ranking': scores_chunk
            })
            result_table_chunk: pa.Table = pa.Table.from_pandas(result_chunk_df, schema=relevant_extended_schema, preserve_index=False)
            writer.write_table(result_table_chunk)
            processed_pairs_count += len(result_chunk_df)

            del relevant_chunk_df, merged_chunk_df, scores_chunk, result_chunk_df, result_table_chunk

        writer.close()
        print(f"\nPomyślnie utworzono plik: {output_path}")
        print(f"Przetworzono i zapisano łącznie {processed_pairs_count} par pytanie-dokument.")

        if processed_pairs_count > 0:
            print("\nPodgląd pierwszych 5 wierszy pliku relevant_with_score.parquet:")
            print(pd.read_parquet(output_path).head())
        else:
            print("Nie przetworzono żadnych par, plik wyjściowy może być pusty lub nie zawierać danych.")

    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono jednego z plików wejściowych: {e}. Upewnij się, że pliki istnieją w podanych ścieżkach.")
    except KeyError as e:
        print(f"BŁĄD: Brak oczekiwanej kolumny w jednym z plików Parquet: {e}. Sprawdź strukturę plików wejściowych.")
    except AttributeError as e:
        if 'reranker' in str(e) and hasattr(e, 'obj') and e.obj is None:
            print(f"BŁĄD: Obiekt 'reranker' nie został poprawnie zainicjalizowany (jest None). Szczegóły: {e}")
        elif 'reranker' in str(e):
            print(f"BŁĄD: Problem z obiektem 'reranker'. Upewnij się, że jest poprawnie załadowany i ma metodę 'predict'. Szczegóły: {e}")
        else:
            print(f"BŁĄD: Wystąpił problem z atrybutem: {e}")
    except Exception as e:
        import traceback
        print(f"BŁĄD: Wystąpił nieoczekiwany błąd podczas przetwarzania: {e}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        if 'writer' in locals() and writer is not None and writer.is_open:
            writer.close()
            print("ParquetWriter został zamknięty w bloku finally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring relevant pairs and saving to Parquet.")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"), help="Ścieżka do pliku queries.parquet")
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"), help="Ścieżka do pliku corpus.parquet")
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_PATH"), help="Ścieżka do pliku relevant.parquet")
    parser.add_argument("--output_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"), help="Ścieżka do pliku output.parquet")
    parser.add_argument("--chunk_size", type=int, default=config("PROCESSING_CHUNK_SIZE", cast=int), help="Rozmiar chunku do przetwarzania")
    parser.add_argument("--reranker_batch_size", type=int, default=config("RERANKER_BATCH_SIZE", cast=int), help="Batch size dla rerankera")
    parser.add_argument("--reranker_model_name", type=str, default=config("RERANKER_NAME"), help="Nazwa modelu rerankera")
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