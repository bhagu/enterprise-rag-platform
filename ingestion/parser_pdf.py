"""
parser_pdf.py

PDF parsing module for the RAG ingestion pipeline.

Uses PyMuPDF (fitz) to extract page-level text and metadata
from PDF documents. The output is structured so downstream
components such as the chunking engine can operate easily.
"""

from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF


class PDFParser:
    """
    PDFParser extracts text and metadata from PDF files.
    """

    def parse(self, file_path: str) -> List[Dict]:
        """
        Parse a PDF document and extract page-level text.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            List[Dict]: List of document objects containing
                        page text and metadata.
        """

        parsed_pages = []

        pdf_document = fitz.open(file_path)

        for page_number, page in enumerate(pdf_document):

            text = page.get_text()

            parsed_pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "page": page_number + 1
                    }
                }
            )

        return parsed_pages


def main():
    """
    Manual test for the PDF parser.
    """

    parser = PDFParser()

    sample_pdf = Path(__file__).resolve().parent.parent / "data/raw_documents/Attention_is_All_You_Need.pdf"

    pages = parser.parse(sample_pdf)

    print(f"Parsed {len(pages)} pages\n")

    for page in pages[:2]:
        print(page["metadata"])
        print(page["text"][:200])
        print()


if __name__ == "__main__":
    main()