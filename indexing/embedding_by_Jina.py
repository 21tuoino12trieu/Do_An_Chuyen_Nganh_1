import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv()
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

device = "cuda:0"
MODEL_NAME = "models/jina-embeddings-v3"
ENCODE_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 32
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

items = []
with open(
    "data/Retrieval/semantic_chunking_for_embedding.jsonl",
    "r",
    encoding="utf-8",
) as f:
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

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
model.max_seq_length = 8192

embeddings = model.encode(
    [item["content"] for item in items],
    batch_size=ENCODE_BATCH_SIZE,
    convert_to_numpy=True,
    device=device,
    normalize_embeddings=True,
    show_progress_bar=True,
    task="retrieval.passage",
).astype("float32")

print("Embedding successfully !")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

dim = model.get_sentence_embedding_dimension()
client.create_collection(
    collection_name="legal_clauses_jina-v3",
    vectors_config=qdrant_models.VectorParams(
        size=dim,
        distance=qdrant_models.Distance.DOT,
    ),
    hnsw_config=qdrant_models.HnswConfigDiff(
        m=16,
        ef_construct=200,
        full_scan_threshold=10000,
        max_indexing_threads=0,
        on_disk=False,
    ),
    shard_number=1,
)

total = len(items)
ids = range(total)
batch = []
for idx, (point_id, vector, item) in enumerate(zip(ids, embeddings, items)):
    batch.append(
        qdrant_models.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=item,
        )
    )
    if len(batch) >= UPLOAD_BATCH_SIZE:
        client.upsert(collection_name="legal_clauses_jina-v3", points=batch)
        batch.clear()

client.upsert(collection_name="legal_clauses_jina-v3", points=batch)

print("Upload successfully !")
