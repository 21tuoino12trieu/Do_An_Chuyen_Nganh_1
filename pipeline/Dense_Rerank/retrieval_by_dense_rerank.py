import logging
import os
import re
import unicodedata
from typing import Any, List, Optional, Dict

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import FlagReranker

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_PATH = "models/AITeamVN"

RERANKER_MODEL_PATH = "models/bge-reranker-v2-m3"

logger = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SEARCH_LIMIT = 20


def preprocess_query(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class DenseRerank:
    def __init__(self, collection_name: str):
        if not QDRANT_URL:
            raise EnvironmentError("QDRANT_URL environment variable is required")

        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            device=DEVICE,
        )
        self.reranker = FlagReranker(RERANKER_MODEL_PATH,use_fp16=True)

    def search(self, query: str, top_k: int = DEFAULT_SEARCH_LIMIT) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        prepared_query = preprocess_query(query)
        query_vector = (
            self.embedding_model.encode(
                prepared_query,
                convert_to_numpy=True,
                device=DEVICE,
                normalize_embeddings=True,
            )
            .astype("float32")
            .tolist()
        )
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=None,
            timeout=120,
        )
        payloads = [hit.payload for hit in search_result]

        return payloads
    
    def rerank(
        self,
        query: str,
        top_k: int = DEFAULT_SEARCH_LIMIT,
        candidate_pool: Optional[int] = None,
    ) -> List[dict]:

        pool_size = candidate_pool or DEFAULT_SEARCH_LIMIT
        pool_size = max(pool_size, top_k)

        search_results = self.search(query, top_k=pool_size)
        if not search_results:
            return []
        
        pairs = []
        for search_result in search_results:
            content = search_result["content"]
            pairs.append([query,content])
            
        scores = self.reranker.compute_score(pairs)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        return [
            search_results[idx] for idx in ranked_indices[: min(top_k, len(search_results))]
        ]
