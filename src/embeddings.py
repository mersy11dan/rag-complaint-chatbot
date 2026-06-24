"""Embedding generation and vector store creation."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedder(model_name: str = DEFAULT_MODEL):
    """Return a sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_vector_store(chunks: list[str], embedder, persist_dir: str = "vector_store"):
    """Embed chunks and persist them to a FAISS index on disk."""
    import faiss
    import numpy as np

    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(persist_dir) / "index.faiss"))
    return index
