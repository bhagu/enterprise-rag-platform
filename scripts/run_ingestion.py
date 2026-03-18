"""
run_ingestion.py

End-to-end ingestion pipeline for the Enterprise RAG Platform.

This script:
1. Loads documents
2. Parses content
3. Chunks text
4. Generates embeddings
5. Builds FAISS index
"""
import time
from core.ingestion.document_loader import DocumentLoader
from core.ingestion.parser_pdf import PDFParser
from core.chunking.chunking_engine import ChunkingEngine
from core.embeddings.embedding_service import EmbeddingService
from core.vector_store.faiss_index import FAISSVectorStore

def run_pipeline(data_path: str):
    """
    Run full ingestion pipeline

    Args:
        data_path (str): Path to raw documents
    """

    print("\n🚀 Starting ingestion pipeline...\n")

    # Step 1: Load Documents
    loader = DocumentLoader(data_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents")

    # Step 2: Parse Documents
    parser = PDFParser()
    parsed_pages = []

    for doc in documents:
        for doc in documents:
            if doc["type"] == ".pdf":
                pages = parser.parse(doc["path"])
                parsed_pages.extend(pages)

    print(f"📘 Parsed {len(parsed_pages)} pages")

    # Step 3: Chunk documents
    chunker = ChunkingEngine(chunk_size=300, overlap=50)
    chunks = chunker.chunk_document(parsed_pages)
    print(f"🔹 Generated {len(chunks)} chunks")

    # Step 4: Generate embeddings
    embedder = EmbeddingService()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print("🧠 Generated embeddings")

    # Step 5: Build FAISS index
    dimension = len(chunks_with_embeddings[0]["embedding"])
    vector_store = FAISSVectorStore(dimension)
    vector_store.add_embeddings(chunks_with_embeddings)

    print("📦 FAISS index built successfully")

    return vector_store


def main():
    """
    Entry point for ingestion script.
    """
    start = time.time()
    data_path = "data/raw_documents"

    vector_store = run_pipeline(data_path)

    print("\n✅ Ingestion pipeline completed successfully!\n")
    end = time.time()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    main()
