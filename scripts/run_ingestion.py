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

import json
from pathlib import Path


def load_processed_files(path):
    if path.exists():
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def save_processed_files(path, files):
    with open(path, "w") as f:
        json.dump(list(files), f, indent=2)


def run_pipeline(data_path: str):
    """
    Run full ingestion pipeline

    Args:
        data_path (str): Path to raw documents
    """

    # Load Existing State
    index_path = Path("data/vector_store")
    processed_path = index_path / "processed_files.json"
    processed_files = load_processed_files(processed_path)

    if (index_path / "faiss.index").exists():
        print("📦 Loading existing FAISS index...")
        vector_store = FAISSVectorStore.load(index_path)
    else:
        vector_store = None

    print("\n🚀 Starting ingestion pipeline...\n")

    # Filter New Documents
    loader = DocumentLoader(data_path)
    documents = loader.load_documents()

    new_docs = [
        doc for doc in documents
        if str(doc["path"]) not in processed_files
    ]

    # Skip If Nothing New
    if not new_docs:
        print("✅ No new documents to process")
        return vector_store
      
    

    # Process Only New Docs
    parser = PDFParser()
    parsed_pages = []

    for doc in new_docs:
        if doc["type"] == ".pdf":
            pages = parser.parse(doc["path"])
            parsed_pages.extend(pages)

    print(f"📘 Parsed {len(parsed_pages)} pages")

    # Chunk Documents
    chunker = ChunkingEngine(chunk_size=300, overlap=50)
    chunks = chunker.chunk_document(parsed_pages)
    print(f"🔹 Generated {len(chunks)} chunks")

    # Generate embeddings
    embedder = EmbeddingService()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    print("🧠 Generated embeddings")

    # Initialize Index If Needed
    if vector_store is None:
        dimension = len(chunks_with_embeddings[0]["embedding"])
        vector_store = FAISSVectorStore(dimension)

    # Add to Existing Index
    vector_store.add_embeddings(chunks_with_embeddings)

    print("📦 FAISS index built successfully")

    # Update Processed Files
    for doc in new_docs:
        processed_files.add(str(doc["path"]))

    save_processed_files(processed_path, processed_files)

    # Save Updated Index
    vector_store.save(index_path)
    print("💾 Index updated with new documents")
    
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
