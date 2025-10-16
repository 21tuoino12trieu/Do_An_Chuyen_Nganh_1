import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch 
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


device = "cuda:0"
MODEL_NAME = "/data/small-language-models/cuong/Do_An/model/vietnamese_document_embedding"
ENCODE_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 32

items = []
with open("/data/small-language-models/cuong/Do_An_Chuyen_Nganh_1/data/semantic_chunking.jsonl", "r", encoding="utf-8") as f:
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
        
model = SentenceTransformer(MODEL_NAME,trust_remote_code = True, device=device)
model.max_seq_length = 8096

embeddings = model.encode(
    [item["content"] for item in items],
    batch_size = ENCODE_BATCH_SIZE,
    convert_to_numpy=True,
    device=device, 
    show_progress_bar = True,
).astype("float32")

print("Embedding successfully !")

client = QdrantClient(
    url="https://12eebde4-9fa9-4958-a32c-d0e7779270ae.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qpOUhQvSA7nI0s_qaoPWHOmZU2-tErydTg_6dsp6q6k",
)

dim = model.get_sentence_embedding_dimension()
client.create_collection(
    collection_name="legal_clauses_vn_dcm_embedding",
    vectors_config=qdrant_models.VectorParams(
        size=dim,
        distance=qdrant_models.Distance.COSINE,
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
            collection_name="legal_clauses_vn_dcm_embedding",
            points=batch
        )
        batch.clear()

client.upsert(
    collection_name="legal_clauses_vn_dcm_embedding",
    points=batch
)

print("Upload successfully !")