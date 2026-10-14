
from core.chunking.chunking_engine import ChunkingEngine


def test_invalid_configurations():
    invalid_configs = [
        (0, 0),    # Chunk size must be positive
        (100, -1), # Overlap cannot be negative
        (100, 100),# Overlap must be smaller than chunk size
        (100, 120),# Overlap cannot exceed chunk size
    ]

    for chunk_size, overlap in invalid_configs:
        try:
            ChunkingEngine(
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except ValueError:
            print(
                f"PASS: rejected chunk_size={chunk_size}, "
                f"overlap={overlap}"
            )
        else:
            raise AssertionError(
                f"Invalid configuration accepted: "
                f"chunk_size={chunk_size}, overlap={overlap}"
            )


def test_valid_configuration():
    chunker = ChunkingEngine(chunk_size=100, overlap=20)

    assert chunker.chunk_size == 100
    assert chunker.overlap == 20

    print("PASS: accepted valid chunk configuration")


def test_empty_pages():
    chunker = ChunkingEngine(chunk_size=10, overlap=2)

    parsed_pages = [
        {"text": "", "metadata": {"source": "blank.pdf", "page": 1}},
        {"text": "   \n  ", "metadata": {"source": "blank.pdf", "page": 2}},
    ]

    chunks = chunker.chunk_document(parsed_pages)

    assert chunks == []
    print("PASS: empty pages produce no chunks")


def test_chunking_progress():
    chunker = ChunkingEngine(chunk_size=10, overlap=2)

    parsed_pages = [
        {
            "text": " ".join(f"word{i}" for i in range(25)),
            "metadata": {"source": "sample.pdf", "page": 1},
        }
    ]

    chunks = chunker.chunk_document(parsed_pages)

    assert len(chunks) == 4
    assert chunks[0]["text"].split()[-2:] == ["word8", "word9"]
    assert chunks[1]["text"].split()[:2] == ["word8", "word9"]
    assert chunks[-1]["text"].split()[-1] == "word24"

    print("PASS: overlapping chunks progress and preserve expected boundaries")

if __name__ == "__main__":
    test_invalid_configurations()
    test_valid_configuration()
    test_empty_pages()
    test_chunking_progress()
    print("\nAll chunking configuration tests passed.")
