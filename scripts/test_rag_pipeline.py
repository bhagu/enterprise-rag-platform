from scripts.run_ingestion import run_pipeline
from pipeline.rag_pipeline import RAGPipeline


def main():
    print("\n🚀 Running ingestion...\n")

    vector_store = run_pipeline("data/raw_documents")

    pipeline = RAGPipeline(vector_store)

    query = "What is attention all about?"

    print(f"\n🔍 Query: {query}\n")

    result = pipeline.run(query)

    print("\n💡 Answer:\n")
    print(result["answer"])

    print("\n📚 Sources:\n")
    for src in result["sources"]:
        print(src)


if __name__ == "__main__":
    main()