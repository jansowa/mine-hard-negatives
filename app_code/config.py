from decouple import config


def get_vector_db_backend() -> str:
    backend = config("VECTOR_DB_BACKEND", default="lancedb").strip().lower()
    if backend not in {"qdrant", "lancedb"}:
        raise ValueError("VECTOR_DB_BACKEND must be one of: qdrant, lancedb")
    return backend


def get_qdrant_url() -> str:
    return config("QDRANT_URL", default="http://localhost:6333")


def get_lancedb_path() -> str:
    return config("LANCEDB_PATH", default="./lancedb_data")
