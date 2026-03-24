from core.retrieval.retriever import Retriever
from core.generation.llm_generator import LLMGenerator
from core.reranking.cross_encoder_reranker import CrossEncoderReranker

import time

class RAGPipeline:

    def __init__(self, vector_store):
        self.retriever = Retriever(vector_store)
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()

    def run(self, query: str, top_k: int = 5):

        start = time.time()

        t1 = time.time()
        candidates = self.retriever.retrieve(query, top_k=top_k)
        print(f"Retrieval time: {time.time() - t1:.2f}s")

        t2 = time.time()
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        print(f"Rerank time: {time.time() - t2:.2f}s")

        contexts = [chunk["text"][:500] for _, chunk in reranked]

        t3 = time.time()
        answer = self.generator.generate(query, contexts)
        print(f"Generation time: {time.time() - t3:.2f}s")

        print(f"Total time: {time.time() - start:.2f}s")

        return {
            "answer": answer,
            "sources": [chunk["metadata"] for _, chunk in reranked]
        }