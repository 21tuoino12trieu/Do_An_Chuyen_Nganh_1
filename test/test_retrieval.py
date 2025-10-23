from retrieval_by_dense_vector import DenseVectorRetriever
import os
from dotenv import load_dotenv

load_dotenv()


user_query = input("Enter your query: ")
top_k = int(input("Enter number of top results to retrieve: "))
retriever = DenseVectorRetriever(collection_name="legal_clauses_AITeamVN")
results = retriever.search(query=user_query, top_k=top_k)
for idx, res in enumerate(results):
    print(f"Result {idx+1}: {res}")
