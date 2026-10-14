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
import os
from pathlib import Path

import shutil
from datetime import datetime
import numpy as np

import hashlib

def get_file_timestamp(path: str) -> float:
    """
    Get the last modified timestamp of a file.
    """
    return os.path.getmtime(path)

def load_processed_files(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_processed_files(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_file_hash(file_path: str) -> str:
    """Return the SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def detect_changes(documents, processed_files):
    """
    Detect new, updated, and deleted source documents using SHA-256 hashes.

    Returns:
        tuple: (changes_detected, deleted_files)
    """
    current_files = {
        str(doc["path"]): get_file_hash(str(doc["path"]))
        for doc in documents
    }

    previous_files = set(processed_files.keys())
    current_paths = set(current_files.keys())

    new_files = current_paths - previous_files
    deleted_files = previous_files - current_paths

    updated_files = set()

    for path in current_paths & previous_files:
        previous_value = processed_files[path]

        # Legacy or invalid manifest entries must be refreshed.
        is_valid_hash = (
            isinstance(previous_value, str)
            and len(previous_value) == 64
            and all(
                character in "0123456789abcdefABCDEF"
                for character in previous_value
            )
        )

        if not is_valid_hash or previous_value != current_files[path]:
            updated_files.add(path)
     
    for path in sorted(new_files):
        print(f"New file detected: {path}")

    for path in sorted(updated_files):
        print(f"Updated file detected: {path}")

    for path in sorted(deleted_files):
        print(f"Deleted file detected: {path}")

    changes_detected = bool(
        new_files or updated_files or deleted_files
    )

    return changes_detected, deleted_files



def clear_index(path: Path):
    if path.exists():
        for file in path.glob("*"):
            file.unlink()
        print("🗑️ Old index cleared")




def promote_staged_index(
    index_path: Path,
    staging_path: Path,
    backup_path: Path,
) -> None:
    """
    Promote a validated staged index using directory renames.
    Restore the previous index if promotion fails.
    """
    required_files = [
        "faiss.index",
        "chunks.pkl",
        "processed_files.json",
    ]

    # Validate the staged files before touching the active index.
    for filename in required_files:
        if not (staging_path / filename).is_file():
            raise FileNotFoundError(
                f"Staging file is missing: {filename}"
            )

    staged = FAISSVectorStore.load(staging_path)

    if staged.index.ntotal == 0:
        raise ValueError("Staged index is empty.")

    if staged.index.ntotal != len(staged.chunks):
        raise ValueError(
            "Staged index validation failed: "
            "vector count does not match chunk count."
        )

    # Preserve previous backups; create a unique backup path.
    if backup_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_path.with_name(
            f"{backup_path.name}_{timestamp}"
        )
    had_active_index = index_path.exists()
    previous_index_moved = False

    try:
        # Move the active index out of the way.
        if index_path.exists():
            index_path.rename(backup_path)
            previous_index_moved = True

        # Promote the complete staged directory.
        staging_path.rename(index_path)

        # Verify the promoted index.
        promoted = FAISSVectorStore.load(index_path)

        if promoted.index.ntotal == 0:
            raise ValueError("Promoted index is empty.")

        if promoted.index.ntotal != len(promoted.chunks):
            raise ValueError(
                "Promoted index validation failed."
            )

        print("✅ Staged index promoted successfully")

    
    except Exception as promotion_error:
        try:
            if previous_index_moved:
                if not backup_path.exists():
                    raise FileNotFoundError(
                        f"Cannot restore index: backup missing at {backup_path}"
                    )

                # Move the failed promotion aside instead of deleting it.
                if index_path.exists():
                    failed_path = index_path.with_name(
                        f"{index_path.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    )
                    index_path.rename(failed_path)

                backup_path.rename(index_path)

            elif not had_active_index and index_path.exists():
                failed_path = index_path.with_name(
                    f"{index_path.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                )
                index_path.rename(failed_path)

        except Exception as rollback_error:
            raise RuntimeError(
                f"Promotion failed: {promotion_error}. "
                f"Rollback also failed: {rollback_error}. "
                "Preserve the backup and inspect the index directories manually."
            ) from rollback_error

        raise


def run_pipeline(data_path: str):
    """
    Run ingestion pipeline with FULL REBUILD on any change.
    """

    index_path = Path("data/vector_store")
    staging_path = Path("data/vector_store_staging")
    backup_path = Path("data/vector_store_backup")
    processed_path = index_path / "processed_files.json"

    loader = DocumentLoader(data_path)

    # This pipeline currently supports PDF ingestion only
    documents = [
        doc
        for doc in loader.load_documents()
        if doc["type"] == ".pdf"
    ]

    
    if not documents:
        raise ValueError(
            "No PDF documents found. "
            "Aborting rebuild to protect the existing index."
        )

    processed_files = load_processed_files(processed_path)

    # -------------------------------
    # Step 1: Detect changes
    # -------------------------------
    changes_detected, deleted_files = detect_changes(
        documents,
        processed_files
    )
    # -------------------------------
    # Step 2: No changes → load index
    # -------------------------------
    index_exists = (index_path / "faiss.index").is_file()
    chunks_exist = (index_path / "chunks.pkl").is_file()
    processed_exists = (index_path / "processed_files.json").is_file()


    if not changes_detected and index_exists and chunks_exist and processed_exists:
        print("📦 No changes detected. Loading existing index...")
        return FAISSVectorStore.load(index_path)

    print("\n🚀 Changes detected. Rebuilding index...\n")

    # -------------------------------
    # Step 4: Parse ALL documents
    # -------------------------------
    parser = PDFParser()
    parsed_pages = []

    for doc in documents:
        if doc["type"] == ".pdf":            
            pages = parser.parse(doc["path"])

            if not pages:
                raise ValueError(
                    f"No pages parsed from PDF: {doc['path']}. "
                    "Aborting rebuild to protect the existing index."
                )

            # A PDF can contain pages but still have no extractable text.
            usable_pages = [
                page for page in pages
                if page.get("text", "").strip()
            ]

            if not usable_pages:
                raise ValueError(
                    f"No extractable text found in PDF: {doc['path']}. "
                    "The PDF may be scanned or image-only. "
                    "Aborting rebuild to protect the existing index."
                )

            empty_page_count = len(pages) - len(usable_pages)

            if empty_page_count:
                print(
                    f"⚠️ {doc['path']}: "
                    f"{empty_page_count} page(s) contain no extractable text."
                )

            parsed_pages.extend(usable_pages)


    print(f"📘 Parsed {len(parsed_pages)} pages")

    # -------------------------------
    # Step 5: Chunk
    # -------------------------------
    chunker = ChunkingEngine(chunk_size=300, overlap=50)
    chunks = chunker.chunk_document(parsed_pages)
    print(f"🔹 Generated {len(chunks)} chunks")

    # -------------------------------
    # Step 6: Embed
    # -------------------------------
    embedder = EmbeddingService()
    
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    if not chunks_with_embeddings:
        raise ValueError(
            "No embeddings generated. "
            "Aborting rebuild to protect the existing index."
        )
    
    if len(chunks_with_embeddings) != len(chunks):
        raise ValueError(
            f"Chunk/embedding count mismatch: "
            f"{len(chunks)} chunks but "
            f"{len(chunks_with_embeddings)} embedded chunks. "
            "Aborting rebuild to protect the existing index."
        )

    
    expected_dimension = len(chunks_with_embeddings[0]["embedding"])

    if expected_dimension == 0:
        raise ValueError("Embedding dimension is zero.")

    for i, chunk in enumerate(chunks_with_embeddings):
        embedding = np.asarray(chunk.get("embedding"), dtype=np.float32)

        if embedding.ndim != 1:
            raise ValueError(
                f"Embedding {i} is not a one-dimensional vector."
            )

        if len(embedding) != expected_dimension:
            raise ValueError(
                f"Embedding {i} has dimension {len(embedding)}; "
                f"expected {expected_dimension}."
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                f"Embedding {i} contains NaN or infinite values."
            )

    print(
        f"🧠 Validated {len(chunks_with_embeddings)} embeddings "
        f"with dimension {expected_dimension}"
    )

    # -------------------------------
    # Prepare staging after validation
    # -------------------------------
    if staging_path.exists():
        shutil.rmtree(staging_path)

    staging_path.mkdir(parents=True, exist_ok=True)

    print("🧹 Staging directory prepared")
    
    # -------------------------------
    # Step 7: Build index
    # -------------------------------
    dimension = len(chunks_with_embeddings[0]["embedding"])
    vector_store = FAISSVectorStore(dimension)
    vector_store.add_embeddings(chunks_with_embeddings)

    print("📦 FAISS index built successfully")

    # -------------------------------
    # Step 8: Save index to staging
    # -------------------------------
    vector_store.save(staging_path)
    print("💾 New index saved to staging")

    # Validate the staged index
    staged_index = FAISSVectorStore.load(staging_path)

    if staged_index.index.ntotal != len(staged_index.chunks):
        raise ValueError(
            "Staged index validation failed: "
            "vector count does not match chunk count."
        )

    if staged_index.index.ntotal == 0:
        raise ValueError(
            "Staged index validation failed: index is empty."
        )

    print(
        f"✅ Staged index validated: "
        f"{staged_index.index.ntotal} vectors, "
        f"{len(staged_index.chunks)} chunks"
    )
    # -------------------------------
    # Step 9: Save timestamps
    # -------------------------------
    new_processed = {
        str(doc["path"]): get_file_hash(str(doc["path"]))
        for doc in documents
    }

    staged_processed_path = staging_path / "processed_files.json"
    save_processed_files(staged_processed_path, new_processed)

    # Promote the validated index with backup and rollback
    promote_staged_index(
        index_path=index_path,
        staging_path=staging_path,
        backup_path=backup_path,
    )

    return FAISSVectorStore.load(index_path)

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
