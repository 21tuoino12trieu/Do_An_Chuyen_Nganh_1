import json
from sentence_transformers import SentenceTransformer
import numpy as np
import torch 

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
model.max_seq_length = 2048 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corpus = []

with open("data/semantic_chunking.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            corpus.append(json.loads(line)["content"])

embeddings = model.encode(corpus,device=device,batch_size = 16,convert_to_numpy=True, dtype=np.float32)

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:
    QdrantClient = None
    qdrant_models = None
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
DATA_PATH = Path("data/semantic_chunking.jsonl")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_clauses")
ENCODE_BATCH_SIZE = 64
UPLOAD_BATCH_SIZE = 256
def load_chunks(path: Path) -> List[Dict[str, str]]:
    """Read clause-level records from JSONL and keep required metadata."""
    items: List[Dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(encoding="utf-8") as infile:
        for line in infile:
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
    if not items:
        raise ValueError(f"No usable records were loaded from {path}")
    return items
def encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode texts into float32 embeddings, batching to avoid OOM."""
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=model.device,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)
def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """Create (or recreate) the Qdrant collection with the proper schema."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
def upload_batches(
    client: QdrantClient,
    embeddings: np.ndarray,
    payloads: List[Dict[str, str]],
) -> None:
    """Upload embeddings and payloads to Qdrant in manageable batches."""
    total = len(payloads)
    ids = range(total)
    batch: List[qdrant_models.PointStruct] = []
    for idx, (point_id, vector, payload) in enumerate(
        zip(ids, embeddings, payloads)
    ):
        batch.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )
        if len(batch) >= UPLOAD_BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch.clear()
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading data from {DATA_PATH} ...")
    records = load_chunks(DATA_PATH)
    texts = [record["content"] for record in records]
    print(f"Loading embedding model {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = encode_texts(model, texts)
    vector_size = embeddings.shape[1]
    print(f"Generated {len(embeddings)} embeddings of size {vector_size}.")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_api_key:
        if QdrantClient is None or qdrant_models is None:
            raise RuntimeError(
                "qdrant-client is not installed. Install it with "
                "`pip install qdrant-client` before uploading."
            )
        print(f"Connecting to Qdrant at {qdrant_url} ...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        ensure_collection(client, vector_size)
        print(f"Uploading data to collection '{COLLECTION_NAME}' ...")
        upload_batches(client, embeddings, records)
        print("Upload completed.")
    else:
        # Persist locally so results can be inspected or uploaded later.
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        np.save(output_dir / "embeddings.npy", embeddings)
        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            "Environment variables for Qdrant not found. "
            "Saved embeddings to data/embeddings.npy and metadata to "
            "data/metadata.jsonl instead."
        )
if __name__ == "__main__":
    main()