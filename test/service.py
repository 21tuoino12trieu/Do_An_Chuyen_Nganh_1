from fastapi import FastAPI
from retrieval_by_dense_vector import DenseVectorRetriever
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

search = DenseVectorRetriever(collection_name="legal_clauses_AITeamVN")

HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVICE_PORT", "8683"))


@app.get("/")
def root():
    return {
        "message": "Service is running. Use /api/search/?query=...&top_k=... to query.",
        "docs_url": "/docs",
    }


@app.get("/api/search/")
def search_startup(query: str, top_k: int = 5):
    results = search.search(query=query, top_k=top_k)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
