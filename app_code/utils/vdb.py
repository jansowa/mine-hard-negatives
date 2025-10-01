from qdrant_client import QdrantClient
import os
from decouple import config
import warnings
import logging

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Suppress HTTP request logs from Qdrant client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_qdrant_client():
    """
    Returns the Qdrant client instance.
    """
    return QdrantClient(url=config("QDRANT_URL", "http://qdrantapp-neuca-custom-deployment:6333"))
