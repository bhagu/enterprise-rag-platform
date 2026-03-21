"""
llm_generator.py

Handles LLM-based answer generation using retrieved context.
"""

import subprocess
from typing import List


class LLMGenerator:
    """
    Uses local LLM (Ollama) to generate responses.
    """

    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Build structured prompt for RAG.

        Args:
            query (str)
            contexts (List[str])

        Returns:
            str: formatted prompt
        """

        context_block = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts[:3])]
        )

        prompt = f"""
You are an AI assistant helping answer questions based ONLY on the provided context.

If the answer is not present in the context, say:
"I don't have enough information to answer that."

---------------------
CONTEXT:
{context_block}
---------------------

QUESTION:
{query}

ANSWER (concise, grounded in context):
"""
        return prompt

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer using LLM.

        Args:
            query (str)
            contexts (List[str])

        Returns:
            str
        """

        prompt = self.build_prompt(query, contexts)

        response = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )

        return response.stdout.strip()