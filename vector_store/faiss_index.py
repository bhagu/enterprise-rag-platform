"""
faiss_index.py

Handles vector storage and similarity search using FAISS.
"""

from typing import List, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """
    Manages FAISS index for storing and retrieving embeddings.
    """

    def __init__(self, dimension: int):
        """
        Initialize FAISS index.

        Args:
            dimension (int): Embedding dimension (must match model)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []  # stores original chunk data

    def add_embeddings(self, chunks: List[dict]) -> None:
        """
        Add embeddings to FAISS index.

        Args:
            chunks (List[dict]): List of chunk objects with embeddings
        """

        embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")

        # Normalize embeddings (important!)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        Perform similarity search.

        Args:
            query_embedding (List[float]): Query embedding
            top_k (int): Number of results to return

        Returns:
            List[Tuple[score, chunk]]
        """

        query_vector = np.array([query_embedding]).astype("float32")

        # Normalize query
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for score, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            results.append((score, chunk))

        return results


def main():
    """
    Simple test for FAISS vector store.
    """

    # Dummy chunks
    chunks = [
        {"text": "Transformers use self-attention", "embedding": np.random.rand(384)},
        {"text": "FAISS is used for vector search", "embedding": np.random.rand(384)},
        {"text": "Neural networks learn patterns", "embedding": np.random.rand(384)},
    ]

    store = FAISSVectorStore(dimension=384)
    store.add_embeddings(chunks)

    # Dummy query
    query_embedding = np.random.rand(384)

    results = store.search(query_embedding, top_k=2)

    for score, chunk in results:
        print(f"Score: {score:.4f}")
        print(f"Text: {chunk['text']}\n")


if __name__ == "__main__":
    main()