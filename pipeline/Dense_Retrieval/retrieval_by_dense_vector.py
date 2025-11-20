import logging
import os
import re
import unicodedata
from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
logger = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    logger.warning("CUDA device not available, falling back to CPU for embeddings")

EMBEDDING_CONFIGS: Dict[str, Dict[str, object]] = {
    "legal_clauses_AITeamVN": {
        "model_path": "models/AITeamVN",
        "max_seq_length": 2048,
        "encode_kwargs": {
            "normalize_embeddings": True,
        },
    },
    "legal_clauses_jina-v3": {
        "model_path": "/data/small-language-models/cuong/models/jina-embeddings-v3",
        "model_kwargs": {"trust_remote_code": True},
        "max_seq_length": 8192,
        "encode_kwargs": {
            "normalize_embeddings": True,
            "task": "retrieval.query",
            "prompt_name": "retrieval.query",
        },
    },
    "legal_clauses_Qwen3": {
        "model_path": "/data/small-language-models/cuong/models/Qwen3-Embedding-0.6B",
        "max_seq_length": 32000,
        "encode_kwargs": {
            "normalize_embeddings": True,
            "prompt_name": "query",
        },
    },
    "legal_clauses_vn_dcm_embedding": {
        "model_path": "/data/small-language-models/cuong/models/vietnamese-document-embedding",
        "model_kwargs": {"trust_remote_code": True},
        "max_seq_length": 8096,
        "encode_kwargs": {},
    },
}
EMBEDDING_ALIASES = {
    "AITeamVN": "legal_clauses_AITeamVN",
    "jina-embeddings-v3": "legal_clauses_jina-v3",
    "Qwen3": "legal_clauses_Qwen3",
    "vn_dcm_embedding": "legal_clauses_vn_dcm_embedding",
}

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
    def __init__(self, collection_name: str, embedding_key: Optional[str] = None):
        if not QDRANT_URL:
            logger.error("QDRANT_URL environment variable is required")
            raise EnvironmentError("QDRANT_URL environment variable is required")

        self.collection_name = collection_name

        config_name = embedding_key or collection_name
        config_name = EMBEDDING_ALIASES.get(config_name, config_name)
        self.embedding_config = EMBEDDING_CONFIGS.get(config_name)
        if not self.embedding_config:
            logger.error("No embedding configuration found for key '%s'", config_name)
            raise ValueError(f"No embedding configuration found for key '{config_name}'")

        model_path = self.embedding_config["model_path"]
        if not isinstance(model_path, str) or not os.path.exists(model_path):
            logger.error("Embedding model path is invalid: %s", model_path)
            raise FileNotFoundError(f"Embedding model path is invalid: {model_path}")

        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        except Exception as exc:
            logger.error("Failed to create Qdrant client", exc_info=exc)
            raise

        model_kwargs = dict(self.embedding_config.get("model_kwargs", {}))
        model_kwargs.setdefault("device", DEVICE)

        try:
            self.embedding_model = SentenceTransformer(model_path, **model_kwargs)
        except Exception as exc:
            logger.error("Failed to load embedding model from %s", model_path, exc_info=exc)
            raise

        max_seq_length = self.embedding_config.get("max_seq_length")
        if isinstance(max_seq_length, int):
            self.embedding_model.max_seq_length = max_seq_length

        self.encode_kwargs = dict(self.embedding_config.get("encode_kwargs", {}))
        self.encode_kwargs.setdefault("convert_to_numpy", True)
        self.encode_kwargs.setdefault("device", DEVICE)

        logger.info(
            "DenseVectorRetriever initialized for collection %s (embedding key: %s) on device %s",
            self.collection_name,
            config_name,
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
            query_vector = self.embedding_model.encode(
                normalized_query, **self.encode_kwargs
            ).tolist()
        except Exception as exc:
            logger.error("Failed to encode query for collection %s", self.collection_name, exc_info=exc)
            raise

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=None,
                timeout = 60,
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
