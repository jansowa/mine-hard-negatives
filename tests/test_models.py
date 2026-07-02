import sys
import types
from pathlib import Path

import sentence_transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

import models
from models import get_dense_model, get_reranker_model, rerank


def test_dense_model_always_enables_trust_remote_code(monkeypatch):
    calls = []
    fake_torch = types.SimpleNamespace(bfloat16=object())

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self._client = types.SimpleNamespace(tokenizer=types.SimpleNamespace(model_max_length=None))

    monkeypatch.setattr(models, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)

    get_dense_model("sdadas/stella-pl-retrieval-mini-8k", batch_size=3, gpu_id=2, prompt="query: ")

    assert calls == [
        {
            "model_name": "sdadas/stella-pl-retrieval-mini-8k",
            "model_kwargs": {
                "model_kwargs": {
                    "torch_dtype": fake_torch.bfloat16,
                    "device_map": "cuda:2",
                },
                "trust_remote_code": True,
            },
            "encode_kwargs": {"batch_size": 3, "prompt": "query: "},
        }
    ]


def test_non_flag_rerankers_use_generic_cross_encoder_path(monkeypatch):
    calls = []
    identity = object()
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        nn=types.SimpleNamespace(Identity=lambda: identity),
    )

    def fake_cross_encoder(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return object()

    monkeypatch.setattr(models, "_import_torch", lambda: fake_torch)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "768")

    tokenizer, model = get_reranker_model("example-org/custom-reranker")

    assert tokenizer is None
    assert model is not None
    assert calls == [
        (
            "example-org/custom-reranker",
            {
                "device": "cpu",
                "trust_remote_code": True,
                "model_kwargs": {},
                "activation_fn": identity,
                "max_length": 768,
            },
        )
    ]


def test_flag_embedding_reranker_receives_correct_max_length(monkeypatch):
    calls = []

    class FakeFlagAutoReranker:
        @classmethod
        def from_finetuned(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        types.SimpleNamespace(FlagAutoReranker=FakeFlagAutoReranker),
    )
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "768")

    get_reranker_model("BAAI/bge-reranker-v2-m3")

    assert calls == [
        (
            "BAAI/bge-reranker-v2-m3",
            {
                "use_bf16": True,
                "devices": ["cuda:0"],
                "max_length": 768,
            },
        )
    ]


def test_lightweight_bge_reranker_uses_bf16_flagembedding_path(monkeypatch):
    calls = []

    class FakeFlagAutoReranker:
        @classmethod
        def from_finetuned(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        types.SimpleNamespace(FlagAutoReranker=FakeFlagAutoReranker),
    )

    get_reranker_model("BAAI/bge-reranker-v2.5-gemma2-lightweight")

    assert calls == [
        (
            "BAAI/bge-reranker-v2.5-gemma2-lightweight",
            {
                "use_bf16": True,
                "devices": ["cuda:0"],
            },
        )
    ]


def test_rerank_uses_predict_models():
    class PredictModel:
        def predict(self, pairs, batch_size, show_progress_bar):
            assert pairs == [("q1", "d1"), ("q2", "d2")]
            assert batch_size == 7
            assert show_progress_bar is False
            return [1.25, 2.5]

    scores = rerank(
        tokenizer=None,
        model=PredictModel(),
        query=["q1", "q2"],
        answers=["d1", "d2"],
        batch_size=7,
        model_name="mixedbread-ai/mxbai-rerank-base-v2",
    )

    assert scores == [1.25, 2.5]
