import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.Dense_Retrieval.retrieval_by_dense_vector import DenseVectorRetriever

load_dotenv()


def main():
    user_query = input("Enter your query: ")
    try:
        top_k_input = input("Enter number of top results to retrieve (default 5): ")
        top_k = int(top_k_input) if top_k_input.strip() else 5
    except ValueError:
        print("Invalid input for top_k. Using default value of 5.")
        top_k = 5

    retriever = DenseVectorRetriever(collection_name="legal_clauses_AITeamVN")
    results = retriever.search(query=user_query, top_k=top_k)
    
    for idx, res in enumerate(results):
        answer = f"{res.get('article_id')}#{res.get('clause_id')}"
        content = res.get('content')
        print(f"Result {idx+1}: {answer} {content}")


if __name__ == "__main__":
    main()
