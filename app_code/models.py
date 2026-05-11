from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from decouple import config

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


def _import_torch():
    import torch

    return torch

try:
    from tensorrt import TensorRTDenseEmbeddings, TensorRTReranker, is_tensorrt_model_path
except ImportError:
    TensorRTDenseEmbeddings = None
    TensorRTReranker = None

    def is_tensorrt_model_path(_model_name: str) -> bool:
        return False


# langchain-qdrant is optional in some environments. Keep the imported classes
# and local fallback classes under separate names so mypy does not see a redefinition.
SparseEmbeddingsBase: type[Any]
SparseVectorFactory: type[Any]


class SparseVectorLike(Protocol):
    indices: list[int]
    values: list[float]


try:
    from langchain_qdrant.sparse_embeddings import SparseEmbeddings as ImportedSparseEmbeddings
    from langchain_qdrant.sparse_embeddings import SparseVector as ImportedSparseVector
except ImportError:

    class FallbackSparseEmbeddings:
        pass

    @dataclass
    class FallbackSparseVector:
        indices: list[int]
        values: list[float]

    SparseEmbeddingsBase = FallbackSparseEmbeddings
    SparseVectorFactory = FallbackSparseVector
else:
    SparseEmbeddingsBase = ImportedSparseEmbeddings
    SparseVectorFactory = ImportedSparseVector


class SpladeEmbedding(SparseEmbeddingsBase):
    def __init__(
        self, model_name, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int, default=16), gpu_id: int = 0
    ):
        torch = _import_torch()
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, device_map=f"cuda:{gpu_id}", torch_dtype=torch.float16
        )
        self.model.eval()
        self.device = torch.device(f"cuda:{gpu_id}")
        self.batch_size = batch_size

    def _encode_splade_batch(self, texts: list[str]) -> list[SparseVectorLike]:
        torch = _import_torch()
        inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=config("SPARSE_EMBEDDER_MAX_LENGTH", cast=int, default=None),
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits, attention_mask = outputs["logits"], inputs["attention_mask"]
        attention_mask = attention_mask.unsqueeze(-1)
        vectors = torch.max(torch.log(torch.add(torch.relu(logits), 1)) * attention_mask, dim=1)[0]

        results = []
        for vector in vectors:
            idx = torch.nonzero(vector).squeeze().cpu().numpy().tolist()
            values = vector[idx].cpu().numpy().tolist()
            results.append(SparseVectorFactory(indices=idx, values=values))

        torch.cuda.empty_cache()
        return results

    def embed_query(self, query: str) -> SparseVectorLike:
        return self._encode_splade_batch([query])[0]

    def embed_documents(self, texts: list[str]) -> list[SparseVectorLike]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results.extend(self._encode_splade_batch(batch))
        return results


def get_sparse_model(
    model_name: str, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int, default=16), gpu_id: int = 0
) -> SpladeEmbedding:
    return SpladeEmbedding(model_name, batch_size=batch_size, gpu_id=gpu_id)


def get_dense_model(
    model_name: str, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int, default=16), gpu_id: int = 0, prompt=""
) -> HuggingFaceEmbeddings:
    torch = _import_torch()
    from langchain_huggingface import HuggingFaceEmbeddings

    if is_tensorrt_model_path(model_name):
        return TensorRTDenseEmbeddings(model_name, batch_size=batch_size, prompt=prompt, gpu_id=gpu_id)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "model_kwargs": {
                "torch_dtype": torch.bfloat16,
                "device_map": f"cuda:{gpu_id}",
            }
        },
        encode_kwargs={"batch_size": batch_size, "prompt": prompt},
    )
    embeddings._client.tokenizer.model_max_length = config("DENSE_EMBEDDER_MAX_LENGTH", cast=int, default=None)
    return embeddings


def is_flag_embedding_reranker(model_name: str) -> bool:
    return model_name.startswith("BAAI")


def is_llm_lightweight_reranker(model_name: str) -> bool:
    return model_name.endswith("lightweight")


def get_reranker_model(
    model_name: str = config("RERANKER_NAME", default="cross-encoder/ms-marco-MiniLM-L-6-v2"), gpu_id: int = 0
):
    if is_tensorrt_model_path(model_name):
        return None, TensorRTReranker(
            model_name,
            batch_size=config("RERANKER_BATCH_SIZE", cast=int, default=16),
            gpu_id=gpu_id,
        )

    if is_flag_embedding_reranker(model_name):
        from FlagEmbedding import FlagAutoReranker

        devices = [f"cuda:{gpu_id}"]
        model = FlagAutoReranker.from_finetuned(
            model_name, use_bf16=True, devices=devices, max_legnth=config("RERANKER_MAX_LENGTH")
        )
        # model = FlagAutoReranker.from_finetuned(model_name, use_fp16=True, devices='cpu')
        return None, model

    torch = _import_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Using {torch.cuda.device_count()} GPUs for reranker model")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model


def rerank(
    tokenizer,
    model,
    query: str | Sequence[str],
    answers: list[str],
    batch_size=16,
    model_name: str = config("RERANKER_NAME", default="cross-encoder/ms-marco-MiniLM-L-6-v2"),
) -> list[float]:
    if isinstance(query, str):
        texts = [[query, answer] for answer in answers]
    else:
        texts = [[q, answer] for q, answer in zip(query, answers)]

    if hasattr(model, "score_pairs"):
        return model.score_pairs([(q, a) for q, a in texts], batch_size=batch_size)

    if is_flag_embedding_reranker(model_name):
        results = []
        additional_params: dict[str, Any] = {}
        if is_llm_lightweight_reranker(model_name):
            additional_params["cutoff_layers"] = [28]
            additional_params["compress_ratio"] = 2
            additional_params["compress_layers"] = [24, 40]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            results += model.compute_score(batch_texts, **additional_params)
        return results

    torch = _import_torch()
    result_batches: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding="longest",
            max_length=config("RERANKER_MAX_LENGTH", cast=int, default=None),
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            output = model(**tokens)
        batch_results = output.logits.detach().cpu().float().numpy()
        result_batches.append(batch_results)

    result_array = np.concatenate(result_batches, axis=0)
    result_array = np.squeeze(result_array)
    return [float(result) for result in result_array.tolist()]
