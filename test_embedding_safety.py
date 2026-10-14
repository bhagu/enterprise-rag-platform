
from unittest.mock import patch

from core.embeddings.embedding_service import EmbeddingService


def test_embedding_count_mismatch():
    service = EmbeddingService.__new__(EmbeddingService)

    # Simulate a model that returns only one embedding
    # even though two chunks were supplied.
    class FakeModel:
        def encode(self, texts, show_progress_bar=False):
            return [[0.1, 0.2, 0.3]]

    service.model = FakeModel()

    chunks = [
        {"text": "First test chunk"},
        {"text": "Second test chunk"},
    ]

    try:
        with patch(
            "core.embeddings.embedding_service.tqdm",
            side_effect=lambda iterable: iterable,
        ):
            service.embed_chunks(chunks, batch_size=64)
    except ValueError as exc:
        assert "Embedding count mismatch" in str(exc)
        print("PASS: embedding-count mismatch detected")
        print(f"Expected error: {exc}")
    else:
        raise AssertionError(
            "Expected ValueError for an embedding-count mismatch"
        )


if __name__ == "__main__":
    test_embedding_count_mismatch()
