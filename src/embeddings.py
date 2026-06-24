"""Embedding generation and vector store creation for the complaint RAG project.

Encodes complaint chunks with ``sentence-transformers/all-MiniLM-L6-v2`` and
persists them — together with rich, per-chunk metadata — into a vector store
under ``vector_store/``. Supports both ChromaDB (default) and FAISS backends.

The module is intentionally dependency-light at import time: heavy libraries
(``sentence_transformers``, ``chromadb``, ``faiss``) are imported lazily inside
the functions that need them, so importing this module never fails.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.chunking import Chunk

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "vector_store"
DEFAULT_COLLECTION = "complaints"
DEFAULT_BATCH_SIZE = 64

# Metadata stored alongside every chunk in the vector store.
METADATA_FIELDS = (
    "complaint_id",
    "product_category",
    "product",
    "issue",
    "company",
    "state",
    "date_received",
    "chunk_index",
    "total_chunks",
)

# Chroma only accepts scalar metadata values of these types.
_SCALAR_TYPES = (str, int, float, bool)


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #

def get_embedder(model_name: str = DEFAULT_MODEL):
    """Load and return the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts(
    texts: Sequence[str],
    embedder: Any = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode a list of texts into a 2-D float32 embedding matrix.

    Embeddings are L2-normalized by default so cosine similarity reduces to a
    dot product (which both FAISS inner-product and Chroma cosine rely on).
    """
    if embedder is None:
        embedder = get_embedder(model_name)
    embeddings = embedder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embeddings, dtype="float32")


def encode_chunks(chunks: Sequence[Chunk], embedder: Any = None, **kwargs: Any) -> np.ndarray:
    """Encode the text of each ``Chunk`` into embeddings."""
    return encode_texts([c.text for c in chunks], embedder=embedder, **kwargs)


# --------------------------------------------------------------------------- #
# Metadata assembly
# --------------------------------------------------------------------------- #

def _scalar(value: Any) -> Any:
    """Coerce a value into a Chroma-safe scalar (str/int/float/bool)."""
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        # Guard against NaN floats which break some stores
        if isinstance(value, float) and value != value:  # NaN check
            return ""
        return value
    return str(value)


def build_metadata(chunk: Chunk, total_chunks: int) -> dict:
    """Build the full metadata dict for a chunk.

    Pulls the standard fields from the chunk's own attributes and looks up the
    remaining source fields (issue, company, state, etc.) from ``chunk.metadata``.
    """
    md = chunk.metadata or {}
    record = {
        "complaint_id": _scalar(chunk.complaint_id),
        "product_category": _scalar(md.get("product_category", chunk.product)),
        "product": _scalar(chunk.product),
        "issue": _scalar(md.get("issue", "")),
        "company": _scalar(md.get("company", "")),
        "state": _scalar(md.get("state", "")),
        "date_received": _scalar(md.get("date_received", "")),
        "chunk_index": _scalar(chunk.chunk_index),
        "total_chunks": _scalar(total_chunks),
    }
    return record


def _build_metadatas(chunks: Sequence[Chunk]) -> list[dict]:
    """Build metadata for every chunk, computing total_chunks per complaint."""
    per_complaint = Counter(c.complaint_id for c in chunks)
    return [build_metadata(c, per_complaint[c.complaint_id]) for c in chunks]


def _build_ids(chunks: Sequence[Chunk]) -> list[str]:
    """Deterministic, unique ids: ``<complaint_id>_<chunk_index>_<position>``."""
    return [f"{c.complaint_id}_{c.chunk_index}_{i}" for i, c in enumerate(chunks)]


# --------------------------------------------------------------------------- #
# Vector store: ChromaDB
# --------------------------------------------------------------------------- #

def build_chroma_store(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray | None = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    embedder: Any = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = True,
):
    """Build and persist a ChromaDB collection from chunks + metadata."""
    import chromadb

    if not chunks:
        raise ValueError("No chunks provided to build the vector store.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    if embeddings is None:
        embeddings = encode_chunks(chunks, embedder=embedder, model_name=model_name, batch_size=batch_size)

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=_build_ids(chunks),
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=_build_metadatas(chunks),
    )
    return collection


def load_chroma_store(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Open an existing persisted ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


# --------------------------------------------------------------------------- #
# Vector store: FAISS
# --------------------------------------------------------------------------- #

def build_faiss_store(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray | None = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    model_name: str = DEFAULT_MODEL,
    embedder: Any = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Build a FAISS index (inner-product / cosine) and persist it with metadata.

    Writes two files into ``persist_dir``:
      - ``index.faiss``    — the FAISS index
      - ``metadata.json``  — documents + metadata aligned to index row order
    """
    import faiss

    if not chunks:
        raise ValueError("No chunks provided to build the vector store.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    if embeddings is None:
        embeddings = encode_chunks(chunks, embedder=embedder, model_name=model_name, batch_size=batch_size)

    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(Path(persist_dir) / "index.faiss"))

    payload = {
        "ids": _build_ids(chunks),
        "documents": [c.text for c in chunks],
        "metadatas": _build_metadatas(chunks),
    }
    with open(Path(persist_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return index


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #

def build_vector_store(
    chunks: Sequence[Chunk],
    backend: str = "chroma",
    persist_dir: str = DEFAULT_PERSIST_DIR,
    **kwargs: Any,
):
    """Build and persist a vector store using the chosen backend.

    Args:
        chunks: The chunks to index.
        backend: ``"chroma"`` (default) or ``"faiss"``.
        persist_dir: Directory to persist the store into (defaults to ``vector_store/``).
    """
    backend = backend.lower()
    if backend == "chroma":
        return build_chroma_store(chunks, persist_dir=persist_dir, **kwargs)
    if backend == "faiss":
        return build_faiss_store(chunks, persist_dir=persist_dir, **kwargs)
    raise ValueError(f"Unknown backend {backend!r}; expected 'chroma' or 'faiss'.")
