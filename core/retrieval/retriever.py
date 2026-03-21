"""
retriever.py

Handles query embedding and retrieval of relevant chunks
from FAISS vector store.
"""

from typing import List, Tuple

from core.embeddings.embedding_service import EmbeddingService
from core.vector_store.faiss_index import FAISSVectorStore

import numpy as np

def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Adding MMR Function
    
def mmr_selection(query_embedding, candidates, top_k=5, lambda_param=0.7):
    """
    Apply Max Marginal Relevance selection.
    """

    selected = []
    selected_indices = set()

    # Extract embeddings
    embeddings = [chunk["embedding"] for _, chunk in candidates]

    # Step 1: pick best initial
    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    first_idx = int(np.argmax(scores))

    selected.append(candidates[first_idx])
    selected_indices.add(first_idx)

    # Step 2: iterative selection
    while len(selected) < top_k:

        mmr_scores = []

        for i, emb in enumerate(embeddings):
            if i in selected_indices:
                continue

            relevance = cosine_similarity(query_embedding, emb)

            redundancy = max(
                cosine_similarity(emb, embeddings[j])
                for j in selected_indices
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            mmr_scores.append((mmr_score, i))

        if not mmr_scores:
            break

        _, next_idx = max(mmr_scores, key=lambda x: x[0])

        selected.append(candidates[next_idx])
        selected_indices.add(next_idx)

    return selected



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
  

    def retrieve(self, query: str, top_k: int = 5):

        # Step 1: Query embedding
        query_embedding = self.embedder.model.encode([query])[0]

        # Step 2: Get more candidates
        candidates = self.vector_store.search(query_embedding, top_k * 3)
        
        # Step 3: Apply MMR
        selected = mmr_selection(query_embedding, candidates, top_k=top_k)

        return selected