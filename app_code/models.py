from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from decouple import config
from typing import Tuple
import numpy as np
import torch

class SpladeEmbedding(SparseEmbeddings):
    def __init__(self, model_name, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int)):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                          device_map='cuda', torch_dtype=torch.float16)
        self.model.eval()
        self.device = torch.device("cuda")
        self.batch_size = batch_size

    def _encode_splade_batch(self, texts: list[str]) -> list[SparseVector]:
        inputs = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", max_length=config("EMBEDDER_MAX_LENGTH", cast=int, default=None)).to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits, attention_mask = outputs["logits"], inputs["attention_mask"]
        attention_mask = attention_mask.unsqueeze(-1)
        vectors = torch.max(torch.log(torch.add(torch.relu(logits), 1)) * attention_mask, dim=1)[0]

        results = []
        for vector in vectors:
            idx = torch.nonzero(vector).squeeze().cpu().numpy().tolist()
            values = vector[idx].cpu().numpy().tolist()
            results.append(SparseVector(indices=idx, values=values))

        torch.cuda.empty_cache()
        return results

    def embed_query(self, query: str) -> SparseVector:
        return self._encode_splade_batch([query])[0]

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self._encode_splade_batch(batch))
        return results


def get_sparse_model(model_name: str, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int)) -> SpladeEmbedding:
    return SpladeEmbedding(model_name, batch_size=batch_size)


def get_dense_model(model_name: str, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int),
                    prompt="") -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs={'model_kwargs': {'torch_dtype': torch.bfloat16}},
                                       encode_kwargs={'batch_size': batch_size, 'prompt': prompt})
    embeddings._client.tokenizer.model_max_length = config("EMBEDDER_MAX_LENGTH", cast=int, default=None)
    return embeddings


def is_flag_embedding_reranker(model_name: str) -> bool:
    return model_name.startswith("BAAI")


def is_llm_lightweight_reranker(model_name: str) -> bool:
    return model_name.endswith("lightweight")


def get_reranker_model(model_name: str = config("RERANKER_NAME")):
    if is_flag_embedding_reranker(model_name):
        from FlagEmbedding import FlagAutoReranker
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        model = FlagAutoReranker.from_finetuned(model_name, use_bf16=True, devices=devices, max_legnth=config("RERANKER_NAME"))
        # model = FlagAutoReranker.from_finetuned(model_name, use_fp16=True, devices='cpu')
        return None, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model


def rerank(tokenizer, model, query: Tuple[str, list[str]], answers: list[str], batch_size=16,
           model_name: str = config("RERANKER_NAME")) -> list[float]:
    print("Calculating ranks")
    if isinstance(query, str):
        texts = [[query, answer] for answer in answers]
    else:
        texts = [[q, answer] for q, answer in zip(query, answers)]

    results = []

    if is_flag_embedding_reranker(model_name):
        print("Creating params for FlagEmbedding reranker")
        additional_params = dict()
        if is_llm_lightweight_reranker(model_name):
            additional_params["cutoff_layers"] = [28]
            additional_params["compress_ratio"] = 2
            additional_params["compress_layers"] = [24, 40]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            results += model.compute_score(batch_texts, **additional_params)
        return results

    for i in range(0, len(texts), batch_size):
        print(f"Calculating ranks for batch {i}")
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding="longest",
            max_length=config("RERANKER_MAX_LENGTH", cast=int, default=None),
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        print("Moved tokens to GPU")
        output = model(**tokens)
        print("Calculated reranker output")
        batch_results = output.logits.detach().cpu().float().numpy()
        print("Moved logits to batch_results")
        results.append(batch_results)
        print("Added batch_results to the list")

    results = np.concatenate(results, axis=0)
    results = np.squeeze(results)
    return [float(result) for result in results.tolist()]
