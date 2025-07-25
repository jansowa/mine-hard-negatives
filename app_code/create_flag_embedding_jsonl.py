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
    negatives_multiplication_threshold: float ,
    negatives_subtraction_threshold: float
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
            if negatives_multiplication_threshold is not None:
                threshold = min_positive_score * negatives_multiplication_threshold
            else:
                threshold = min_positive_score  - negatives_subtraction_threshold
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
    parser = argparse.ArgumentParser(description="Creating the jsonl file with positives, negatives and scores.")
    parser.add_argument("--corpus_path", type=str, default=config("CORPUS_PATH"), help="Path to parquet file with corpus")
    parser.add_argument("--queries_path", type=str, default=config("QUERIES_PATH"), help="Path to parquet file with queries")
    parser.add_argument("--relevant_path", type=str, default=config("RELEVANT_WITH_SCORE_PATH"), help="Path to parquet file with relevant connections (with scores)")
    parser.add_argument("--negatives_path", type=str, default=config("NEGATIVES_PATH"), help="Path to parquet file with negatives")
    parser.add_argument("--output_path", type=str, default=config("OUTPUT_PATH"), help="Path to output parquet file")
    parser.add_argument("--num_negatives", type=int, default=config("NUM_NEGATIVES", cast=int), help="Number of negatives for each question")

    negatives_subtraction_threshold_str = config("NEGATIVES_SUBTRACTION_THRESHOLD", default=None)
    if negatives_subtraction_threshold_str is not None and negatives_subtraction_threshold_str != "":
        negatives_subtraction_threshold = float(negatives_subtraction_threshold_str)
    else:
        negatives_subtraction_threshold = None


    negatives_multiplication_threshold_str = config("NEGATIVES_MULTIPLICATION_THRESHOLD", default=None)
    print(f"{negatives_multiplication_threshold_str=}")
    if negatives_multiplication_threshold_str is not None and negatives_multiplication_threshold_str != "":
        negatives_multiplication_threshold = float(negatives_multiplication_threshold_str)
    else:
        negatives_multiplication_threshold = None

    if negatives_subtraction_threshold is not None and negatives_multiplication_threshold is not None:
        raise ValueError("You have to choose between subtraction and multiplication threshold.")

    if negatives_subtraction_threshold is None and negatives_multiplication_threshold is None:
        raise ValueError("You have to fill in the negatives threshold")

    parser.add_argument("--negatives_multiplication_threshold", required=False, type=float, default=negatives_multiplication_threshold, help="Multiplication threshold for negatives (percent of the minimum positive)")
    parser.add_argument("--negatives_subtraction_threshold", required=False, type=float, default=negatives_subtraction_threshold, help="Subtraction threshold for negatives (this value will be subtracted from the minimum positive)")
    args = parser.parse_args()

    process_negatives(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        relevant_path=args.relevant_path,
        negatives_path=args.negatives_path,
        output_path=args.output_path,
        num_negatives=args.num_negatives,
        negatives_multiplication_threshold=args.negatives_multiplication_threshold,
        negatives_subtraction_threshold=args.negatives_subtraction_threshold
    )