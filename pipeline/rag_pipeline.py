from core.retrieval.retriever import Retriever
from core.generation.llm_generator import LLMGenerator
from core.reranking.cross_encoder_reranker import CrossEncoderReranker


class RAGPipeline:

    def __init__(self, vector_store):
        self.retriever = Retriever(vector_store)
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()

    def run(self, query: str, top_k: int = 5):

        # Step 1: Retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=top_k * 3)

        # Step 2: Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        contexts = [chunk["text"] for _, chunk in reranked]

        # Step 3: Generate answer
        answer = self.generator.generate(query, contexts)

        return {
            "answer": answer,
            "sources": [chunk["metadata"] for _, chunk in reranked]
        }