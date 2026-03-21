"""
retrieval_evaluator.py

Evaluates retrieval quality using simple metrics.
"""

from typing import List


class RetrievalEvaluator:
    """
    Evaluates retrieval performance.
    """

    def recall_at_k(self, retrieved_chunks: List[dict], relevant_texts: List[str]) -> float:
        """
        Recall@K = relevant retrieved / total relevant

        Args:
            retrieved_chunks (List[dict])
            relevant_texts (List[str])

        Returns:
            float
        """

        retrieved_texts = [chunk["text"] for _, chunk in retrieved_chunks]

        hits = 0
        for rel in relevant_texts:
            if any(rel in text for text in retrieved_texts):
                hits += 1

        return hits / len(relevant_texts) if relevant_texts else 0.0

    def precision_at_k(self, retrieved_chunks: List[dict], relevant_texts: List[str]) -> float:
        """
        Precision@K = relevant retrieved / total retrieved
        """

        retrieved_texts = [chunk["text"] for _, chunk in retrieved_chunks]

        hits = 0
        for text in retrieved_texts:
            if any(rel in text for rel in relevant_texts):
                hits += 1

        return hits / len(retrieved_texts) if retrieved_texts else 0.0