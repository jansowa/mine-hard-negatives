import pandas as pd
import json
from pathlib import Path

# Parametry wejściowe
num_negatives = 3
negatives_threshold = 0.8

# Ścieżki do plików
data_path = Path("data")
corpus_path = data_path / "corpus.parquet"
queries_path = data_path / "queries.parquet"
relevant_path = data_path / "relevant_with_score.parquet"
negatives_path = data_path / "negatives.parquet"

# Wczytanie danych
corpus_df = pd.read_parquet(corpus_path)
queries_df = pd.read_parquet(queries_path)
relevant_df = pd.read_parquet(relevant_path)
negatives_df = pd.read_parquet(negatives_path)

# Stworzenie słowników pomocniczych
corpus_dict = corpus_df.set_index("id")["text"].to_dict()
queries_dict = queries_df.set_index("id")["text"].to_dict()

# Grupowanie pozytywnych powiązań
positive_grouped = relevant_df.groupby("query_id").agg(
    pos_ids=("document_id", list),
    pos_scores=("positive_ranking", list),
    min_score=("positive_ranking", min)
).reset_index()

# Grupowanie negatywnych powiązań
negatives_grouped = negatives_df.groupby("query_id")

output = []

for _, row in positive_grouped.iterrows():
    qid = row["query_id"]
    query_text = queries_dict.get(qid, "")

    pos_ids = row["pos_ids"]
    pos_scores = row["pos_scores"]
    min_positive_score = row["min_score"]

    pos_texts = [corpus_dict[doc_id] for doc_id in pos_ids]

    # Pobierz negatywy powiązane z tym zapytaniem
    if qid in negatives_grouped.groups:
        neg_rows = negatives_grouped.get_group(qid)
        # Filtruj wg progu
        threshold = negatives_threshold * min_positive_score
        filtered_negs = neg_rows[neg_rows["ranking"] < threshold]
        top_negs = filtered_negs.sort_values(by="ranking", ascending=False).head(num_negatives)

        neg_ids = top_negs["document_id"].tolist()
        neg_scores = top_negs["ranking"].tolist()
        neg_texts = [corpus_dict[doc_id] for doc_id in neg_ids]

        output.append({
            "query": query_text,
            "pos": pos_texts,
            "neg": neg_texts,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "pos_id": pos_ids,
            "neg_id": neg_ids
        })

# Zapis do pliku JSONL
output_path = data_path / "output.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")