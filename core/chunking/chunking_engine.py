"""
chunking_engine.py

Responsible for splitting parsed documents into chunks suitable for
embedding and retrieval.

Implements fixed-size chunking with overlap.
"""

from typing import List, Dict

class ChunkingEngine:
    """
    Splits parsed documents into chunks for embedding
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        """
        Initializes the chunking engine.

        Args:
            chunk_size (int): The size of each chunk in tokens.
            overlap (int): The number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, parsed_pages: Dict) -> List[Dict]:
        """
        Converts parsed pages into chunks.

        Args:
            parsed_pages (Dict): output from parser.

        Returns:
            List[Dict]: A list of chunk objects
        """
        chunks = []
        chunk_id = 0

        for page in parsed_pages:
            words = page["text"].split()

            start = 0
            end = self.chunk_size

            while start < len(words):
                chunk_words =  words[start:end]

                chunk_text = " ".join(chunk_words)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": page["metadata"]
                    }
                )

                chunk_id += 1

                start = end - self.overlap
                end = start + self.chunk_size
        
        return chunks

def main():
    """
    Simple test for the chunking engine
    """

    # Dummy parsed input

    parsed_pages = [
        {
            "text": "Transformers use self attention to model relationships between tokens. " * 50,
            "metadata": {"source": "sample.pdf", "page": 1},
        }
    ]

    chunker = ChunkingEngine(chunk_size=50, overlap=10)
    chunks = chunker.chunk_document(parsed_pages)

    print(f"Generated {len(chunks)} chunks\n")

    for chunk in chunks[:2]:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(chunk["text"][:200])
        print(chunk["metadata"])
        print()

if __name__ == "__main__":
    main() 