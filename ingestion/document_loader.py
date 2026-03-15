"""
document_loader.py

Module responsible for discovering and loading raw documents from the
data directory. The loader scans the input folder and returns metadata
about each document so that downstream components (parsers, chunking
engine, etc.) can process them.

This module does NOT parse document content. It only identifies files
and their types.

Supported file types will later include:
- PDF
- Markdown
- HTML
"""

from pathlib import Path
from typing import List, Dict

class DocumentLoader:
    """
    DocumentLoader scans a directory and identifies documents
    available for ingestion into the RAG pipeline.
    """
    
    def __init__(self, data_path:str) -> None:
        """
        Initialize the document loader.

        Args:
            data_path (str): Path to the directory containing raw documents.
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Document directory not found: {self.data_path}"
            )

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Discover all documents in the data directory.

        Returns:
            List[Dict[str, str]]: A list of document metadata dictionaries.
            Each dictionary contains:
                - path: absolute path to the document
                - type: file extension
        """
        documents: List[Dict[str, str]] = []

        # Iterate through all files in the data directory
        for file_path in self.data_path.glob("*"):
            if file_path.is_file():
                documents.append(
                    {
                        "path": str(file_path),
                        "type": file_path.suffix.lower(),
                    }
                )

        return documents


def main() -> None:
    """
    Example usage for manual testing of the document loader.
    """
    loader = DocumentLoader("../data/raw_documents")
    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents\n")

    for doc in documents:
        print(doc)


if __name__ == "__main__":
    main()