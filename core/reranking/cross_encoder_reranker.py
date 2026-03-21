"""
cross_encoder_reranker.py

Reranks retrieved chunks using a cross-encoder model.
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Uses cross-encoder to rerank retrieved chunks.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, dict]],
        top_k: int = 5
    ) -> List[Tuple[float, dict]]:
        """
        Rerank candidates based on query relevance.

        Args:
            query (str)
            candidates (List[(score, chunk)])
            top_k (int)

        Returns:
            List[(score, chunk)]
        """

        pairs = [(query, chunk["text"]) for _, chunk in candidates]

        scores = self.model.predict(pairs)

        reranked = list(zip(scores, [chunk for _, chunk in candidates]))

        reranked.sort(key=lambda x: x[0], reverse=True)

        return reranked[:top_k]