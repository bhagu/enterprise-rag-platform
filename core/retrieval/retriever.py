"""
retriever.py

Handles query embedding and retrieval of relevant chunks
from FAISS vector store.
"""

from typing import List, Tuple

from core.embeddings.embedding_service import EmbeddingService
from core.vector_store.faiss_index import FAISSVectorStore


class Retriever:
    """
    Converts user queries into embeddings and retrieves
    relevant chunks from the vector store.
    """

    def __init__(self, vector_store: FAISSVectorStore):
        """
        Initialize retriever.

        Args:
            vector_store (FAISSVectorStore): Initialized FAISS store
        """
        self.vector_store = vector_store
        self.embedder = EmbeddingService()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query (str): User query
            top_k (int): Number of results to return

        Returns:
            List[Tuple[float, dict]]: List of (score, chunk)
        """

        # Step 1: Convert query to embedding
        query_embedding = self.embedder.model.encode([query])[0]

        # Step 2: Perform similarity search
        results = self.vector_store.search(query_embedding, top_k)

        return results