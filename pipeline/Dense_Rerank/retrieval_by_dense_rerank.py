import logging
import os
import re
import unicodedata
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_PATH = (
    "/data/small-language-models/cuong/models/AITeamVN"
)

RERANKER_MODEL_PATH = (
    "/data/small-language-models/cuong/models/Qwen3-Reranker-0.6B"
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
        )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output
    
    def process_inputs(self, pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

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
        
