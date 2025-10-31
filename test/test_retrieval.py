import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.Dense_Retrieval.retrieval_by_dense_vector import DenseVectorRetriever

load_dotenv()


user_query = input("Enter your query: ")
top_k = int(input("Enter number of top results to retrieve: "))
retriever = DenseVectorRetriever(collection_name="legal_clauses_AITeamVN")
results = retriever.search(query=user_query, top_k=top_k)
for idx, res in enumerate(results):
    answer = f"{res.get("article_id")}#{res.get("clause_id")}"
    content = res.get("content")
    print(f"Result {idx+1}: {answer} {content}")
