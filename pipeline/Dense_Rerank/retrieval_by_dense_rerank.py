import logging
import os
import re
import unicodedata
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_PATH = (
    "/data/small-language-models/cuong/Do_An/model/AITeamVN/Vietnamese_Embedding"
)

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_query(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class DenseRerank:
    def __init__(self, collection_name: str):
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=httpx.Timeout(connect=10, read=180, write=20, pool=60),
        )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def search(self, query: str, top_k: int = 50):
        query_vector = self.embedding_model.encode(
            query, convert_to_numpy=True, device=DEVICE, normalize_embeddings=True
        ).astype("float32").tolist()
        search_result = self.qdrant_client.search(
            collection_name = "legal_clauses_AITeamVN",
            query_vector = query_vector,
            limit = top_k,
            query_filter = None,
        )
        payloads = [hit.payload for hit in search_result]
        
        return payloads
    
    def rerank(self, List[str]):
        return 
        
