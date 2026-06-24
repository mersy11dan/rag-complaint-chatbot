"""Core Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use only the following retrieved complaint excerpts to formulate your answer.
If the context does not contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class RAGPipeline:
    """Retrieve relevant complaint chunks and generate grounded answers."""

    def __init__(self, retriever, generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for a question."""
        return self.retriever(question, self.top_k)

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(chunk.text for chunk in chunks)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def answer(self, question: str) -> dict:
        """Run retrieval + generation and return the answer with its sources."""
        chunks = self.retrieve(question)
        prompt = self.build_prompt(question, chunks)
        response = self.generator(prompt)
        return {"answer": response, "sources": chunks}
