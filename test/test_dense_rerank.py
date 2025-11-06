import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.Dense_Rerank.retrieval_by_dense_rerank import DenseRerank


load_dotenv()


def main() -> None:
    user_query = input("Enter your query: ")
    top_k_input = input("Enter number of top results to rerank: ")
    try:
        top_k = int(top_k_input)
    except ValueError:
        raise ValueError("top_k must be an integer")

    reranker = DenseRerank(collection_name="legal_clauses_AITeamVN")
    results = reranker.rerank(query=user_query, top_k=top_k)
    for idx, res in enumerate(results, start=1):
        article_id = res.get("article_id") or "N/A"
        clause_id = res.get("clause_id") or "N/A"
        print(f"Result {idx}: {article_id}#{clause_id}")


if __name__ == "__main__":
    main()
