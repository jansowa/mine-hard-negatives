from typing import Tuple
from decouple import config
from models import is_flag_embedding_reranker, is_llm_lightweight_reranker, get_reranker_model
import numpy as np


def rerank(tokenizer, model, query: Tuple[str, list[str]], answers: list[str], old_params: bool, batch_size=16,
           model_name: str = config("RERANKER_NAME")) -> list[float]:
    if isinstance(query, str):
        texts = [[query, answer] for answer in answers]
    else:
        texts = [[q, answer] for q, answer in zip(query, answers)]

    results = []
    if is_flag_embedding_reranker(model_name):
        additional_params = dict()
        if is_llm_lightweight_reranker(model_name):
            if old_params:
                additional_params["cutoff_layers"] = [28]
                additional_params["compress_ratio"] = 2
                additional_params["compress_layers"] = [24, 40]
            else:
                additional_params["cutoff_layers"] = [25]
                additional_params["compress_ratio"] = 2
                additional_params["compress_layers"] = [8]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            results += model.compute_score(batch_texts, **additional_params)
        return results

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding="longest",
            max_length=config("RERANKER_MAX_LENGTH", cast=int, default=None),
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        output = model(**tokens)
        batch_results = output.logits.detach().cpu().float().numpy()
        results.append(batch_results)

    results = np.concatenate(results, axis=0)
    results = np.squeeze(results)
    return [float(result) for result in results.tolist()]


tokenizer, reranker = get_reranker_model()

question = "Dlaczego niebo jest niebieskie?"

answers = [
    "Ponieważ cząsteczki powietrza rozpraszają światło niebieskie bardziej niż inne kolory.",  # bardzo poprawna
    "Bo promienie słoneczne ulegają rozproszeniu w atmosferze.",  # poprawna
    "Bo światło niebieskie najlepiej przechodzi przez powietrze.",  # częściowo poprawna
    "Bo to odbicie koloru oceanu.",  # popularny mit
    "Bo Bóg tak chciał.",  # subiektywna
    "Bo niebieski to kolor spokoju.",  # symboliczna
    "Bo w dzień świeci słońce, a ono jest żółte, więc niebo musi być niebieskie.",  # błędna logika
    "Bo Ziemia obraca się z dużą prędkością.",  # niepoprawna
    "Bo niebo jest zrobione z gazu o kolorze niebieskim.",  # błędna
    "Bo powietrze to właściwie woda w innej formie.",  # błędna
    "Bo satelity malują niebo na niebiesko.",  # absurdalna
    "Bo komputer tak ustawił kolory świata.",  # żartobliwa
    "Bo to efekt odbicia światła od lodowców.",  # błędna
    "Bo tak mówi Wikipedia.",  # neutralna, nie merytoryczna
    "Nie wiem, ale wygląda ładnie.",  # uczciwie niepoprawna
]

old_ranks = rerank(tokenizer, reranker, question, answers, True)
new_ranks = rerank(tokenizer, reranker, question, answers, False)

print("STARY/NOWY WYNIK:")
for old_rank, new_rank in zip(old_ranks, new_ranks):
    print(old_rank, new_rank)