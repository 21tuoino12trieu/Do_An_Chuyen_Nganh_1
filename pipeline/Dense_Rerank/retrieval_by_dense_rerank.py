import logging
import os
import re
import unicodedata
from typing import List, Optional

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_PATH = "/data/small-language-models/cuong/models/AITeamVN"

RERANKER_MODEL_PATH = "/data/small-language-models/cuong/models/Qwen3-Reranker-0.6B"

logger = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SEARCH_LIMIT = 50


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH,
            padding_side="left",
        )
        self.rerank_model = (
            AutoModelForCausalLM.from_pretrained(RERANKER_MODEL_PATH))
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        self.prefix = (
            '<|im_start|>system\n'
            'Bạn là giám khảo pháp lý. Dựa trên câu hỏi và đoạn văn bản được cung cấp, hãy trả lời "yes" nếu đoạn văn trả lời đúng câu hỏi, '
            'và "no" nếu không. Chỉ trả lời "yes" hoặc "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.task = "Cho một câu hỏi pháp luật tiếng Việt, hãy đánh giá đoạn văn bản có phù hợp với câu hỏi hay không."


    def format_instruction(self, instruction, query, doc_payload):
        if instruction is None:
            instruction = "Cho một câu hỏi pháp luật tiếng Việt, hãy đánh giá đoạn văn bản có phù hợp với câu hỏi hay không."
        doc_text = doc_payload.get("content", "")
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc_text}"
    
    def process_inputs(self, pairs):
        encoded = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            return_attention_mask=False,
        )

        for i, token_ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        padded = self.tokenizer.pad(
            encoded,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        for key in padded:
            padded[key] = padded[key].to(self.rerank_model.device)

        return padded
    
    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.rerank_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def search(self, query: str, top_k: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
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

        pairs = [
            self.format_instruction(self.task, query, doc)
            for doc in search_results
        ]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        return [
            search_results[idx] for idx in ranked_indices[: min(top_k, len(search_results))]
        ]
