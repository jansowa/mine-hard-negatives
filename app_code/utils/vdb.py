from qdrant_client import QdrantClient
import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)


def get_qdrant_client():
    """
    Returns the Qdrant client instance.
    """
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://vdb:6333"))
