"""
embedding_service.py

Responsible for generating embeddings for text chunks using
sentence-transformers.

Supports batch processing for efficiency.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingService:
    """
    Handles embedding generation for text chunks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize embedding model.

        Args:
            model_name (str): HuggingFace model name
        """
        self.model = SentenceTransformer(model_name)

    def embed_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 64
    ) -> List[dict]:
        """
        Generate embeddings for chunks in batches.

        Args:
            chunks (List[dict]): List of chunk objects
            batch_size (int): Number of chunks per batch

        Returns:
            List[dict]: Chunks with embeddings added
        """

        texts = [chunk["text"] for chunk in chunks]
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]

            batch_embeddings = self.model.encode(batch, show_progress_bar=False)

            embeddings.extend(batch_embeddings)

        # Attach embeddings back to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks


def main():
    """
    Simple test for embedding service.
    """

    chunks = [
        {"text": "Transformers use self-attention."},
        {"text": "FAISS is used for vector similarity search."}
    ]

    service = EmbeddingService()
    embedded_chunks = service.embed_chunks(chunks)

    for chunk in embedded_chunks:
        print(len(chunk["embedding"]))  # embedding dimension


if __name__ == "__main__":
    main()