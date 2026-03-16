![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/VectorSearch-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Enterprise RAG Platform

An end-to-end Retrieval-Augmented Generation (RAG) system designed to ingest heterogeneous documents, build a vector search index, and enable grounded question answering using local LLMs.

This project demonstrates a **production-style architecture** for building scalable GenAI knowledge systems.

---

# Features

- Document ingestion pipeline
- Multi-format parsing (PDF, Markdown, HTML)
- Token-aware chunking engine
- Embedding generation using sentence-transformers
- FAISS-based vector search
- Retrieval evaluation framework
- FastAPI-based RAG service
- Local LLM inference via Ollama
- Source citations in responses

---

# Architecture

Documents  
↓  
Document Loader  
↓  
Parser Layer (PDF / Markdown / HTML)  
↓  
Chunking Engine  
↓  
Embedding Pipeline  
↓  
FAISS Vector Index  
↓  
Retriever  
↓  
LLM (Ollama - Mistral)  
↓  
Answer + Sources

---

# Project Structure

enterprise-rag-platform/

configs/  
data/  
raw_documents/  
processed_chunks/

ingestion/  
document_loader.py  
parser_pdf.py  
parser_markdown.py  
parser_html.py  

chunking/  
chunking_engine.py  

embeddings/  
embedding_service.py  

vector_store/  
faiss_index.py  

retrieval/  
retriever.py  

evaluation/  
retrieval_metrics.py  

api/  
rag_api.py  

ui/  
streamlit_app.py  

scripts/  
run_ingestion.py  

README.md  
requirements.txt  

---

# Installation

Clone the repository

git clone https://github.com/bhagu/enterprise-rag-platform.git  
cd enterprise-rag-platform  

Create virtual environment

python3.11 -m venv rag-env  

Activate environment (Windows)

rag-env\Scripts\activate  

Install dependencies

pip install -r requirements.txt  

---

# Running Document Ingestion

Add documents to

data/raw_documents/

Run the ingestion pipeline

python scripts/run_ingestion.py

This pipeline will:

1. Load documents
2. Parse document content
3. Generate semantic chunks
4. Create embeddings
5. Build the FAISS vector index

---

# Running the RAG API

Start the API server

uvicorn api.rag_api:app --reload

This exposes an endpoint that allows querying the knowledge base.

---

# Example Query Flow

User Question  
↓  
Query Embedding  
↓  
FAISS Vector Search  
↓  
Top-k Relevant Chunks  
↓  
Context Construction  
↓  
LLM (Mistral via Ollama)  
↓  
Answer + Source Citations  

---

# Dataset

Place documents inside

data/raw_documents/

Supported formats

- PDF
- Markdown
- HTML

Example documents can be stored in

data/raw_documents/example/

Large datasets are intentionally excluded from the repository.

---

# Future Improvements

Planned enhancements:

- Hybrid retrieval (BM25 + vector search)
- Cross-encoder reranking
- Streaming responses
- Retrieval observability metrics
- Real-time document ingestion
- Vector index sharding for scale

---

# Learning Goals

This project is part of a deeper learning journey focused on:

- Retrieval engineering
- LLM systems design
- scalable GenAI infrastructure
- evaluation-driven RAG development

---

# Author

Bhagyaraj Chitoth

Building toward **Principal / Staff AI Engineer (GenAI Systems)** through deep work in:

- Retrieval-Augmented Generation
- vector search infrastructure
- large-scale LLM systems
- AI platform architecture