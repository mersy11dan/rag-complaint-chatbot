"""Retrieval-Augmented Generation pipeline for the complaint chatbot.

Wires together three pieces:
  1. A **Retriever** that embeds a question with the SAME model used at indexing
     time and returns the top-k most relevant chunks from the persisted vector store.
  2. A **prompt template** that constrains the LLM to answer ONLY from the retrieved
     context, and to say so when the context is insufficient.
  3. A **Generator** that combines the prompt + retrieved chunks + question into an answer.

Heavy dependencies (chromadb, faiss, transformers, sentence-transformers) are imported
lazily, and the LLM / embedder / store are all injectable, so the core logic is easy to
unit-test without those libraries installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.embeddings import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
    encode_texts,
    get_embedder,
)

DEFAULT_TOP_K = 5

NO_CONTEXT_MESSAGE = (
    "I don't have enough information in the provided complaint excerpts to answer "
    "that question."
)

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to \
answer questions about customer complaints using ONLY the retrieved complaint excerpts \
provided below.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain enough information to answer, reply exactly: \
"{no_context}"
- Be concise and factual. Do not invent details or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store, with its similarity score."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure helpers (no heavy dependencies — fully unit-testable)
# --------------------------------------------------------------------------- #

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, source-tagged context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        product = chunk.metadata.get("product") or chunk.metadata.get("product_category", "")
        company = chunk.metadata.get("company", "")
        tag = " | ".join(p for p in (product, company) if p)
        header = f"[Source {i}{f' — {tag}' if tag else ''}]"
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    template: str = PROMPT_TEMPLATE,
) -> str:
    """Build the final LLM prompt from the question and retrieved chunks."""
    context = format_context(chunks)
    return template.format(
        context=context,
        question=question,
        no_context=NO_CONTEXT_MESSAGE,
    )


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

class Retriever:
    """Embed a question and fetch the top-k chunks from the persisted store."""

    def __init__(
        self,
        store: Any = None,
        backend: str = "chroma",
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        embedder: Any = None,
    ):
        self.backend = backend.lower()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self._embedder = embedder
        self._store = store

    # -- lazy resources ----------------------------------------------------- #

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.model_name)
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            self._store = self._load_store()
        return self._store

    def _load_store(self):
        if self.backend == "chroma":
            from src.embeddings import load_chroma_store

            return load_chroma_store(self.persist_dir, self.collection_name)
        if self.backend == "faiss":
            import faiss

            index = faiss.read_index(str(Path(self.persist_dir) / "index.faiss"))
            with open(Path(self.persist_dir) / "metadata.json", encoding="utf-8") as f:
                payload = json.load(f)
            return {"index": index, "payload": payload}
        raise ValueError(f"Unknown backend {self.backend!r}; expected 'chroma' or 'faiss'.")

    # -- query -------------------------------------------------------------- #

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a single question with the same model used at indexing time."""
        return encode_texts(
            [question],
            embedder=self.embedder,
            model_name=self.model_name,
            normalize=True,
            show_progress_bar=False,
        )[0]

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for a question."""
        if not question or not question.strip():
            return []
        query_vec = self.embed_query(question)
        if self.backend == "chroma":
            return self._retrieve_chroma(query_vec, top_k)
        return self._retrieve_faiss(query_vec, top_k)

    def _retrieve_chroma(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        result = self.store.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        # Chroma returns cosine distance; convert to a similarity score
        return [
            RetrievedChunk(text=doc, score=1.0 - dist, metadata=meta or {})
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def _retrieve_faiss(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        index = self.store["index"]
        payload = self.store["payload"]
        scores, idxs = index.search(query_vec.reshape(1, -1).astype("float32"), top_k)
        chunks = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunks.append(
                RetrievedChunk(
                    text=payload["documents"][idx],
                    score=float(score),
                    metadata=payload["metadatas"][idx] or {},
                )
            )
        return chunks

    # Allow using the retriever as a plain callable: retriever(question, k)
    def __call__(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        return self.retrieve(question, top_k)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

def get_default_llm(model_name: str = "google/flan-t5-base") -> Callable[[str], str]:
    """Return a simple text-generation callable backed by a HuggingFace pipeline."""
    from transformers import pipeline

    pipe = pipeline("text2text-generation", model=model_name, max_new_tokens=256)

    def _generate(prompt: str) -> str:
        return pipe(prompt)[0]["generated_text"].strip()

    return _generate


class Generator:
    """Turn a question + retrieved chunks into a grounded answer."""

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        template: str = PROMPT_TEMPLATE,
        model_name: str = "google/flan-t5-base",
    ):
        self._llm = llm
        self.template = template
        self.model_name = model_name

    @property
    def llm(self) -> Callable[[str], str]:
        if self._llm is None:
            self._llm = get_default_llm(self.model_name)
        return self._llm

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer; short-circuit to a safe message when no context."""
        if not chunks:
            return NO_CONTEXT_MESSAGE
        prompt = build_prompt(question, chunks, self.template)
        return self.llm(prompt)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class RAGPipeline:
    """End-to-end retrieve-then-generate pipeline."""

    def __init__(
        self,
        retriever: Retriever | Callable,
        generator: Generator,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        """Retrieve context, generate an answer, and return it with its sources."""
        chunks = self.retriever(question, self.top_k)
        answer = self.generator.generate(question, chunks)
        return {"question": question, "answer": answer, "sources": chunks}


def build_rag_pipeline(
    backend: str = "chroma",
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_MODEL,
    llm: Callable[[str], str] | None = None,
    llm_model: str = "google/flan-t5-base",
    top_k: int = DEFAULT_TOP_K,
) -> RAGPipeline:
    """Convenience factory that builds a ready-to-use pipeline from a persisted store."""
    retriever = Retriever(
        backend=backend,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=embedding_model,
    )
    generator = Generator(llm=llm, model_name=llm_model)
    return RAGPipeline(retriever, generator, top_k=top_k)
