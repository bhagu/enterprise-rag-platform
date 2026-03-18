"""
test_retrieval.py

Test script for retrieval pipeline.
"""

from scripts.run_ingestion import run_pipeline
from core.retrieval.retriever import Retriever


def main():
    print("\n🚀 Running ingestion pipeline...\n")

    vector_store = run_pipeline("data/raw_documents")

    retriever = Retriever(vector_store)

    query = "What is FAISS used for?"

    print(f"\n🔍 Query: {query}\n")

    results = retriever.retrieve(query, top_k=3)

    for score, chunk in results:
        print(f"Score: {score:.4f}")
        print(f"Text: {chunk['text'][:200]}")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 50)


if __name__ == "__main__":
    main()