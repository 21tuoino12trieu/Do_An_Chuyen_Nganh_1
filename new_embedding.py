import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch 
from pathlib import Path
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
ENCODE_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 256

items = []
with open("data/semantic_chunking.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        content = record.get("content", "").strip()
        if not content:
            continue    
        items.append(
            {
                "article_id": record.get("article_id"),
                "clause_id": record.get("clause_id"),
                "content": content,
            }
        )

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    [item["content"] for item in items],
    batch_size = ENCODE_BATCH_SIZE,
    convert_to_numpy=True,
    device=device, 
    dtype=np.float32,
    normalize_embeddings=True,
)

client = QdrantClient(
    url="https://12eebde4-9fa9-4958-a32c-d0e7779270ae.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qpOUhQvSA7nI0s_qaoPWHOmZU2-tErydTg_6dsp6q6k",
)

total = len(items)
ids = range(total)
batch = []
for idx, (point_id, vector, item) in enumerate(
    zip(ids, embeddings, items)
):
    batch.append(
        qdrant_models.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=item,
        )
    )
    if len(batch) >= UPLOAD_BATCH_SIZE:
        client.upsert(
            collection_name="legal_clauses",
            points=batch
        )
        batch.clear()

client.upsert(
    collection_name="legal_clauses",
    points=batch
)