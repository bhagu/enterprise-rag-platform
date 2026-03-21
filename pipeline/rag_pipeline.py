from core.retrieval.retriever import Retriever
from core.generation.llm_generator import LLMGenerator


class RAGPipeline:

    def __init__(self, vector_store):
        self.retriever = Retriever(vector_store)
        self.generator = LLMGenerator()

    def run(self, query: str, top_k: int = 5) -> dict:

        results = self.retriever.retrieve(query, top_k)

        contexts = [chunk["text"] for _, chunk in results]

        answer = self.generator.generate(query, contexts)

        return {
            "answer": answer,
            "sources": [chunk["metadata"] for _, chunk in results]
        }