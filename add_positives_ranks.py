from decouple import config
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from models import get_reranker_model, rerank

QUERIES_PATH = 'data/queries.parquet'
CORPUS_PATH = 'data/corpus.parquet'
RELEVANT_PATH = 'data/relevant.parquet'
OUTPUT_PATH = 'data/relevant_with_score.parquet'
OUTPUT_DIR = os.path.dirname(OUTPUT_PATH)

CHUNK_SIZE = config("PROCESSING_CHUNK_SIZE", default=100000, cast=int)
RERANKER_BATCH_SIZE = config("RERANKER_BATCH_SIZE", default=32, cast=int)
RERANKER_MODEL_NAME = config("RERANKER_NAME")


print("Rozpoczynam tworzenie pliku relevant_with_score.parquet...")

try:
    print(f"Wczytywanie danych z:\n - {QUERIES_PATH} (kolumny: id, text)\n - {CORPUS_PATH} (kolumny: id, text)")
    queries_df = pd.read_parquet(QUERIES_PATH, columns=['id', 'text'])
    queries_df = queries_df.rename(columns={'text': 'query_text'}).set_index(
        'id')

    corpus_df = pd.read_parquet(CORPUS_PATH, columns=['id', 'text'])
    corpus_df = corpus_df.rename(columns={'text': 'document_text'}).set_index(
        'id')
    print("Pliki queries.parquet i corpus.parquet wczytane i przygotowane.")

    tokenizer, reranker = get_reranker_model()
    print(f"Reranker załadowany. Urządzenie: {reranker.device}")

    relevant_extended_schema = pa.schema([
        ('query_id', pa.int32()),
        ('document_id', pa.int32()),
        ('positive_ranking', pa.float32())
    ])
    print("Zdefiniowano schemat dla pliku wyjściowego.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Utworzono katalog docelowy: {OUTPUT_DIR}")

    writer = pq.ParquetWriter(OUTPUT_PATH, relevant_extended_schema)
    print(f"Rozpoczęto zapis do pliku: {OUTPUT_PATH}")

    print(f"Przetwarzanie pliku {RELEVANT_PATH} w chunkach o rozmiarze {CHUNK_SIZE}...")
    parquet_file_relevant = pq.ParquetFile(RELEVANT_PATH)
    total_batches = parquet_file_relevant.num_row_groups

    processed_pairs_count = 0

    # Użyj iter_batches z pyarrow do wczytywania chunków bez wczytywania całego pliku
    # Wybieramy tylko potrzebne kolumny od razu
    for i, batch in enumerate(
            tqdm(parquet_file_relevant.iter_batches(batch_size=CHUNK_SIZE, columns=['query_id', 'document_id']),
                 desc="Przetwarzanie chunków relevant.parquet")):
        relevant_chunk_df = batch.to_pandas()
        print(f"Przetwarzanie chunka {i+1} ({len(relevant_chunk_df)} wierszy)...")

        if relevant_chunk_df.empty:
            continue

        merged_chunk_df = pd.merge(
            relevant_chunk_df,
            queries_df,
            left_on='query_id',
            right_index=True,
            how='inner'
        )

        merged_chunk_df = pd.merge(
            merged_chunk_df,
            corpus_df,
            left_on='document_id',
            right_index=True,
            how='inner'
        )

        if merged_chunk_df.empty:
            continue

        scores_chunk = rerank(tokenizer, reranker, merged_chunk_df['query_text'].values.tolist(),
                              merged_chunk_df['document_text'].values.tolist(), batch_size=RERANKER_BATCH_SIZE)

        result_chunk_df = pd.DataFrame({
            'query_id': merged_chunk_df['query_id'],
            'document_id': merged_chunk_df['document_id'],
            'positive_ranking': scores_chunk
        })
        result_table_chunk = pa.Table.from_pandas(result_chunk_df, schema=relevant_extended_schema,
                                                  preserve_index=False)
        writer.write_table(result_table_chunk)
        processed_pairs_count += len(result_chunk_df)

        del relevant_chunk_df, merged_chunk_df, scores_chunk, result_chunk_df, result_table_chunk

    writer.close()
    print(f"\nPomyślnie utworzono plik: {OUTPUT_PATH}")
    print(f"Przetworzono i zapisano łącznie {processed_pairs_count} par pytanie-dokument.")

    if processed_pairs_count > 0:
        print("\nPodgląd pierwszych 5 wierszy pliku relevant_with_score.parquet:")
        print(pd.read_parquet(OUTPUT_PATH).head())
    else:
        print("Nie przetworzono żadnych par, plik wyjściowy może być pusty lub nie zawierać danych.")


except FileNotFoundError as e:
    print(
        f"BŁĄD: Nie znaleziono jednego z plików wejściowych: {e}. Upewnij się, że pliki istnieją w podanych ścieżkach.")
except KeyError as e:
    print(f"BŁĄD: Brak oczekiwanej kolumny w jednym z plików Parquet: {e}. Sprawdź strukturę plików wejściowych.")
except AttributeError as e:
    if 'reranker' in str(e) and hasattr(e, 'obj') and e.obj is None:
        print(f"BŁĄD: Obiekt 'reranker' nie został poprawnie zainicjalizowany (jest None). Szczegóły: {e}")
    elif 'reranker' in str(e):
        print(
            f"BŁĄD: Problem z obiektem 'reranker'. Upewnij się, że jest poprawnie załadowany i ma metodę 'predict'. Szczegóły: {e}")
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