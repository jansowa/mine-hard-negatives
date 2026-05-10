import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_code"))

from config import get_vector_db_backend


def test_get_vector_db_backend_accepts_supported_values(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_BACKEND", "lancedb")
    assert get_vector_db_backend() == "lancedb"

    monkeypatch.setenv("VECTOR_DB_BACKEND", "qdrant")
    assert get_vector_db_backend() == "qdrant"


def test_get_vector_db_backend_rejects_invalid(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_BACKEND", "bad")
    try:
        get_vector_db_backend()
        assert False, "Expected ValueError"
    except ValueError:
        pass
