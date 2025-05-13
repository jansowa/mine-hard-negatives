from sentence_transformers import CrossEncoder
from decouple import config
import torch

def load_reranker(model_name=config("RERANKER_NAME")) -> CrossEncoder:
    return CrossEncoder(
        model_name,
        default_activation_function=torch.nn.Identity(),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

reranker = load_reranker()

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

QUERIES_PATH = 'data/queries.parquet'
CORPUS_PATH = 'data/corpus.parquet'
RELEVANT_PATH = 'data/relevant.parquet'
OUTPUT_PATH = 'data/relevant_with_score.parquet'
OUTPUT_DIR = os.path.dirname(OUTPUT_PATH)

print("Rozpoczynam tworzenie pliku relevant-extended.parquet...")

try:
    print(f"Wczytywanie danych z:\n - {QUERIES_PATH}\n - {CORPUS_PATH}\n - {RELEVANT_PATH}")
    queries_df = pd.read_parquet(QUERIES_PATH)
    corpus_df = pd.read_parquet(CORPUS_PATH)
    relevant_df = pd.read_parquet(RELEVANT_PATH)
    print("Pliki Parquet wczytane pomyślnie.")

    print("Łączenie danych...")
    merged_df = pd.merge(
        relevant_df,
        queries_df,
        left_on='query_id',
        right_on='id',
        how='inner'
    )
    merged_df = merged_df.rename(columns={'text': 'query_text'})
    merged_df = merged_df.drop(columns=['id'])

    merged_df = pd.merge(
        merged_df,
        corpus_df,
        left_on='document_id',
        right_on='id',
        how='inner'
    )
    merged_df = merged_df.rename(columns={'text': 'document_text'})
    merged_df = merged_df.drop(columns=['id'])
    print(f"Dane połączone. Znaleziono {len(merged_df)} par pytanie-dokument do oceny.")

    if merged_df.empty:
        print("Nie znaleziono pasujących par pytanie-dokument. Plik wyjściowy nie zostanie utworzony.")
    else:
        print("Przygotowywanie danych wejściowych dla rerankera...")
        queries_list = merged_df['query_text'].tolist()
        documents_list = merged_df['document_text'].tolist()

        reranker_input = [[query, document] for query, document in zip(queries_list, documents_list)]
        print(f"Przygotowano {len(reranker_input)} par tekstów.")

        print("Obliczanie wyników za pomocą rerankera...")
        scores = reranker.predict(reranker_input, batch_size=config("RERANKER_BATCH_SIZE", cast=int)).tolist()
        print("Wyniki obliczone.")

        merged_df['positive_ranking'] = scores
        print("Dodano kolumnę 'positive_ranking'.")

        relevant_extended_df = merged_df[['query_id', 'document_id', 'positive_ranking']]

        relevant_extended_schema = pa.schema([
            ('query_id', pa.int32()),
            ('document_id', pa.int32()),
            ('positive_ranking', pa.float32())
        ])
        print("Zdefiniowano schemat dla pliku wyjściowego.")

        relevant_extended_table = pa.Table.from_pandas(relevant_extended_df, schema=relevant_extended_schema)
        print("Skonwertowano DataFrame do tabeli PyArrow.")

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Utworzono katalog docelowy: {OUTPUT_DIR}")

        print(f"Zapisywanie wyniku do: {OUTPUT_PATH}")
        pq.write_table(relevant_extended_table, OUTPUT_PATH)
        print(f"Pomyślnie utworzono plik: {OUTPUT_PATH}")

        print("\nPodgląd pierwszych 5 wierszy pliku relevant-extended.parquet:")
        print(pd.read_parquet(OUTPUT_PATH).head())

except FileNotFoundError as e:
    print(f"BŁĄD: Nie znaleziono jednego z plików wejściowych: {e}. Upewnij się, że pliki istnieją w podanych ścieżkach.")
except KeyError as e:
     print(f"BŁĄD: Brak oczekiwanej kolumny w jednym z plików Parquet: {e}. Sprawdź strukturę plików wejściowych.")
except AttributeError as e:
    if 'reranker' in str(e):
        print(f"BŁĄD: Problem z obiektem 'reranker'. Upewnij się, że jest poprawnie załadowany i ma metodę 'predict'. Szczegóły: {e}")
    else:
        print(f"BŁĄD: Wystąpił problem z atrybutem: {e}")
except Exception as e:
    print(f"BŁĄD: Wystąpił nieoczekiwany błąd podczas przetwarzania: {e}")