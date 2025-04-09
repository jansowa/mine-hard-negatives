from models import get_dense_model, get_sparse_model
import argparse
import json
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from sentence_transformers import CrossEncoder
import torch.nn


def find_negatives(dense_model_name: str, sparse_model_name:str, embedding_batch_size: int, collection_name: str, database_path: str, dataset_path: str, output_path: str, top_k: int):
    dense_embeddings = get_dense_model(dense_model_name, batch_size=embedding_batch_size)
    print("Loaded dense embeddings")
    sparse_embeddings = get_sparse_model(sparse_model_name, batch_size=embedding_batch_size)
    print("Loaded sparse embeddings")
    client = QdrantClient(path=database_path)
    print("Loaded database")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )
    reranker = load_reranker()
    print("Loaded reranker")
    with open(dataset_path, 'r', encoding='utf-8') as f, open(output_path, 'a', encoding='utf-8') as out_f:
        for line in f:
            line = line.strip()
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSON error - skipped line: {line}")
                continue
            query = ""
            positive = ""
            for message in obj.get("messages", []):
                if message.get("role") == "user":
                    # TODO: throw error if content is empty!
                    query = message.get("content", "")
                if message.get("role") == "assistant":
                    # TODO: throw error if content is empty!
                    positive = message.get("content", "")
                    retrieved_docs = [document.page_content for document in retriever.get_relevant_documents(query) if document.page_content != positive]
                    retrieved_scores = reranker.predict([[query, positive]] + [[query, retrieved_doc] for retrieved_doc in retrieved_docs],
                                                        batch_size=8).tolist()
                    output_obj = {
                        "query": query,
                        "pos": [positive],
                        "neg": retrieved_docs,
                        "pos_scores": [retrieved_scores[0]],
                        "neg_scores": retrieved_scores[1:]
                    }
                    out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")


def load_reranker(model_name="sdadas/polish-reranker-base-ranknet") -> CrossEncoder:
    return CrossEncoder(
        model_name,
        default_activation_function=torch.nn.Identity(),
        max_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--dense_model_name", type=str, required=False, help="Name of dense model to calculate embeddings", default="sdadas/mmlw-retrieval-roberta-base")
    parser.add_argument("--sparse_model_name", type=str, required=False, help="Name of sparse model to calculate embeddings", default="sdadas/polish-splade")
    parser.add_argument("--embedding_batch_size", type=int, required=False, help="Number of documents in one embeddings model batch", default=16)
    parser.add_argument("--database_collection_name", type=str, required=False, help="Name of database collection", default="all_documents")
    parser.add_argument("--database_path", type=str, required=False, help="Path to the output JSONL file (optional).",
                        default="qdant_db")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--top_k", type=int, default=30, required=False, help="Number of documents to retrieve")
    args = parser.parse_args()
    find_negatives(args.dense_model_name, args.sparse_model_name, args.embedding_batch_size, args.database_collection_name, args.database_path, args.dataset_path, args.output_path, args.top_k)