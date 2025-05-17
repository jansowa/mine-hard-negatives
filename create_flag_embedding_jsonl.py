import pandas as pd
import json
import argparse
from decouple import config

def process_negatives(
    corpus_path: str,
    queries_path: str,
    relevant_path: str,
    negatives_path: str,
    output_path: str,
    num_negatives: int,
    negatives_threshold: float
) -> None:

    corpus_df = pd.read_parquet(corpus_path)
    queries_df = pd.read_parquet(queries_path)
    relevant_df = pd.read_parquet(relevant_path)
    negatives_df = pd.read_parquet(negatives_path)

    corpus_dict = corpus_df.set_index("id")["text"].to_dict()
    queries_dict = queries_df.set_index("id")["text"].to_dict()

    positive_grouped = relevant_df.groupby("query_id").agg(
        pos_ids=("document_id", list),
        pos_scores=("positive_ranking", list),
        min_score=("positive_ranking", min)
    ).reset_index()

    negatives_grouped = negatives_df.groupby("query_id")

    output = []

    for _, row in positive_grouped.iterrows():
        qid = row["query_id"]
        query_text = queries_dict.get(qid, "")

        pos_ids = row["pos_ids"]
        pos_scores = row["pos_scores"]
        min_positive_score = row["min_score"]

        pos_texts = [corpus_dict[doc_id] for doc_id in pos_ids]

        if qid in negatives_grouped.groups:
            neg_rows = negatives_grouped.get_group(qid)
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

    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Zapisano plik: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tworzenie pliku output.jsonl z pozytywami i negatywami.")
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"), help="Ścieżka do corpus.parquet")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"), help="Ścieżka do queries.parquet")
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"), help="Ścieżka do relevant_with_score.parquet")
    parser.add_argument("--negatives_path", type=str, default=config("NEGATIVES_PATH"), help="Ścieżka do negatives.parquet")
    parser.add_argument("--output_path", type=str, default=config("OUTPUT_PATH"), help="Ścieżka z wygenerowanym plikiem JSONL")
    parser.add_argument("--num_negatives", type=int, default=config("NUM_NEGATIVES", cast=int), help="Liczba negatywnych przykładów do wybrania")
    parser.add_argument("--negatives_threshold", type=float, default=config("NEGATIVES_THRESHOLD", cast=float), help="Próg dla negatywów (procent od minimalnego pozytywnego)")
    args = parser.parse_args()

    process_negatives(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        relevant_path=args.relevant_path,
        negatives_path=args.negatives_path,
        output_path=args.output_path,
        num_negatives=args.num_negatives,
        negatives_threshold=args.negatives_threshold
    )