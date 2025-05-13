from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langchain_huggingface import HuggingFaceEmbeddings
from decouple import config
import torch

class SpladeEmbedding(SparseEmbeddings):
    def __init__(self, model_name, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int)):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                          device_map='cuda')
        self.model.eval()
        self.device = torch.device("cuda")
        self.batch_size = batch_size

    def _encode_splade_batch(self, texts: list[str]) -> list[SparseVector]:
        inputs = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", max_length=512).to(
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

def get_dense_model(model_name: str, batch_size: int = config("EMBEDDER_BATCH_SIZE", cast=int), prompt="") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={'batch_size': batch_size, 'prompt': prompt}
    )