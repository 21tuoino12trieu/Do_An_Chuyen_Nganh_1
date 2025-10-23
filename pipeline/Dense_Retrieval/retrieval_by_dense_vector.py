import os
import re
import unicodedata
import logging
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_PATH = "/data/small-language-models/cuong/Do_An/model/AITeamVN/Vietnamese_Embedding"
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    logger.warning("CUDA device not available, falling back to CPU for embeddings")

def preprocess_query(raw_query: str) -> str:
    """Normalize a user query to reduce noise before encoding."""

    if not isinstance(raw_query, str):
        logger.error("Query must be a string, received %s", type(raw_query))
        raise TypeError("Query must be a string")

    trimmed = raw_query.strip()
    if not trimmed:
        logger.warning("Query is empty after trimming whitespace")
        raise ValueError("Query is empty after trimming whitespace")

    normalized = unicodedata.normalize("NFC", trimmed)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"([!?.,])\1+", r"\1", normalized)

    return normalized


class DenseVectorRetriever:
    def __init__(self, collection_name: str):
        if not QDRANT_URL:
            logger.error("QDRANT_URL environment variable is required")
            raise EnvironmentError("QDRANT_URL environment variable is required")

        if not EMBEDDING_MODEL_PATH or not os.path.exists(EMBEDDING_MODEL_PATH):
            logger.error("Embedding model path is invalid: %s", EMBEDDING_MODEL_PATH)
            raise FileNotFoundError("Embedding model path is invalid")

        self.collection_name = collection_name

        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        except Exception as exc:
            logger.error("Failed to create Qdrant client", exc_info=exc)
            raise

        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
        except Exception as exc:
            logger.error("Failed to load embedding model from %s", EMBEDDING_MODEL_PATH, exc_info=exc)
            raise

        logger.info(
            "DenseVectorRetriever initialized for collection %s on device %s",
            self.collection_name,
            DEVICE,
        )

    def search(self, query: str, top_k: int):
        if top_k <= 0:
            logger.error("top_k must be positive, received %s", top_k)
            raise ValueError("top_k must be a positive integer")

        try:
            normalized_query = preprocess_query(query)
        except Exception:
            # preprocess_query already logged details.
            raise

        try:
            query_vector = self.embedding_model.encode(normalized_query).tolist()
        except Exception as exc:
            logger.error("Failed to encode query for collection %s", self.collection_name, exc_info=exc)
            raise

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=None,
            )
        except Exception as exc:
            logger.error(
                "Qdrant search failed for collection %s (top_k=%s)",
                self.collection_name,
                top_k,
                exc_info=exc,
            )
            raise

        payloads = [hit.payload for hit in search_result]
        logger.info(
            "Retrieved %d hits from collection %s (top_k=%d)",
            len(payloads),
            self.collection_name,
            top_k,
        )
        return payloads
