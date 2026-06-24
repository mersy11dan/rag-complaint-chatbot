"""Chunking utilities for complaint narratives.

Splits long complaint narratives into smaller, overlapping chunks suitable for
embedding, while preserving the originating ``complaint_id`` and ``product`` so
each chunk remains traceable back to its source complaint.

Uses LangChain's ``RecursiveCharacterTextSplitter`` when available and falls
back to a dependency-free, word-boundary-aware splitter otherwise.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

DEFAULT_CHUNK_SIZE = 500       # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50     # characters shared between consecutive chunks
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Default CFPB column names (override via function arguments if yours differ)
ID_COL = "Complaint ID"
PRODUCT_COL = "Product"
NARRATIVE_COL = "Consumer complaint narrative"


@dataclass
class Chunk:
    """A single text chunk with its source metadata."""

    text: str
    complaint_id: Any
    product: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flatten the chunk (including extra metadata) into a single dict."""
        data = asdict(self)
        extra = data.pop("metadata")
        return {**data, **extra}


# --------------------------------------------------------------------------- #
# Core splitting
# --------------------------------------------------------------------------- #

def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _build_langchain_splitter(chunk_size: int, chunk_overlap: int, separators: list[str]):
    """Return a LangChain RecursiveCharacterTextSplitter, or None if unavailable."""
    try:  # langchain >= 0.2 split out the text splitters package
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:  # older langchain
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            return None
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )


def _custom_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Word-boundary-aware, character-length-based splitter with overlap.

    This is the dependency-free fallback used when LangChain is not installed.
    Chunks never exceed ``chunk_size`` characters unless a single word is longer
    than ``chunk_size`` (in which case that word becomes its own chunk).
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def joined_len(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + max(0, len(parts) - 1)

    def tail_overlap(parts: list[str]) -> list[str]:
        """Keep trailing words whose combined length stays within the overlap."""
        kept: list[str] = []
        for word in reversed(parts):
            if joined_len([word, *kept]) > chunk_overlap:
                break
            kept.insert(0, word)
        return kept

    for word in words:
        sep = 1 if current else 0
        if current and current_len + sep + len(word) > chunk_size:
            chunks.append(" ".join(current))
            current = tail_overlap(current)
            current_len = joined_len(current)
        sep = 1 if current else 0
        current.append(word)
        current_len += sep + len(word)

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
    use_langchain: bool = True,
) -> list[str]:
    """Split a single text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Characters shared between consecutive chunks.
        separators: Separator hierarchy for the LangChain splitter.
        use_langchain: Prefer LangChain's splitter when it is installed.

    Returns:
        A list of chunk strings (empty list for empty/non-string input).
    """
    _validate(chunk_size, chunk_overlap)
    if not isinstance(text, str) or not text.strip():
        return []

    separators = separators or DEFAULT_SEPARATORS
    if use_langchain:
        splitter = _build_langchain_splitter(chunk_size, chunk_overlap, separators)
        if splitter is not None:
            return [c for c in splitter.split_text(text) if c.strip()]
    return _custom_split(text, chunk_size, chunk_overlap)


# --------------------------------------------------------------------------- #
# Metadata-aware chunking
# --------------------------------------------------------------------------- #

def chunk_narrative(
    text: str,
    complaint_id: Any,
    product: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extra_metadata: dict | None = None,
    **split_kwargs: Any,
) -> list[Chunk]:
    """Split one narrative into ``Chunk`` objects carrying source metadata."""
    pieces = split_text(text, chunk_size, chunk_overlap, **split_kwargs)
    return [
        Chunk(
            text=piece,
            complaint_id=complaint_id,
            product=product,
            chunk_index=i,
            metadata=dict(extra_metadata or {}),
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_dataframe(
    df: pd.DataFrame,
    text_col: str = NARRATIVE_COL,
    id_col: str = ID_COL,
    product_col: str = PRODUCT_COL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **split_kwargs: Any,
) -> list[Chunk]:
    """Chunk every narrative in a DataFrame, preserving id and product metadata.

    Rows whose ``id_col`` or ``product_col`` are missing fall back to ``None`` /
    empty string respectively, so chunking never fails on incomplete rows.
    """
    if text_col not in df.columns:
        raise KeyError(f"DataFrame is missing required column: {text_col!r}")

    n = len(df)
    texts = df[text_col].tolist()
    ids = df[id_col].tolist() if id_col in df.columns else [None] * n
    products = df[product_col].tolist() if product_col in df.columns else [""] * n

    all_chunks: list[Chunk] = []
    for text, complaint_id, product in zip(texts, ids, products):
        product = "" if pd.isna(product) else product
        all_chunks.extend(
            chunk_narrative(
                text=text,
                complaint_id=complaint_id,
                product=product,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **split_kwargs,
            )
        )
    return all_chunks


def chunks_to_dataframe(chunks: list[Chunk]) -> pd.DataFrame:
    """Convert a list of ``Chunk`` objects into a flat DataFrame."""
    return pd.DataFrame([c.to_dict() for c in chunks])


# --------------------------------------------------------------------------- #
# Helper / summary functions
# --------------------------------------------------------------------------- #

def count_chunks(chunks: list[Chunk] | list[str]) -> int:
    """Return the total number of chunks."""
    return len(chunks)


def _chunk_texts(chunks: list[Chunk] | list[str]) -> list[str]:
    return [c.text if isinstance(c, Chunk) else c for c in chunks]


def chunk_length_summary(chunks: list[Chunk] | list[str]) -> dict:
    """Summarize chunk lengths (in characters).

    Returns a dict with the chunk count and min/max/mean/median/total lengths.
    Works on both ``Chunk`` objects and raw strings. Returns zeros when empty.
    """
    texts = _chunk_texts(chunks)
    if not texts:
        return {
            "num_chunks": 0,
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "total_chars": 0,
        }

    lengths = [len(t) for t in texts]
    return {
        "num_chunks": len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
        "total_chars": sum(lengths),
    }
