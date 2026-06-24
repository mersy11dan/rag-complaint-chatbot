"""Tests for src.chunking."""

import pandas as pd
import pytest

from src.chunking import (
    Chunk,
    chunk_dataframe,
    chunk_length_summary,
    count_chunks,
    split_text,
)


# --------------------------------------------------------------------------- #
# split_text
# --------------------------------------------------------------------------- #

def test_split_text_empty_returns_empty_list():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_non_string_returns_empty_list():
    assert split_text(None) == []


def test_split_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("a b c", chunk_size=10, chunk_overlap=10)


def test_split_text_short_text_single_chunk():
    chunks = split_text("a short complaint", chunk_size=100, chunk_overlap=10)
    assert chunks == ["a short complaint"]


def test_custom_split_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = split_text(text, chunk_size=30, chunk_overlap=8, use_langchain=False)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)


def test_custom_split_has_overlap():
    text = " ".join(f"word{i}" for i in range(40))
    chunks = split_text(text, chunk_size=30, chunk_overlap=12, use_langchain=False)
    # The first word of the second chunk should reappear from the first chunk's tail
    assert chunks[1].split()[0] in chunks[0].split()


# --------------------------------------------------------------------------- #
# Metadata preservation
# --------------------------------------------------------------------------- #

def test_chunk_dataframe_preserves_metadata():
    long_text = " ".join(f"token{i}" for i in range(60))
    df = pd.DataFrame(
        {
            "Complaint ID": [101],
            "Product": ["Credit card"],
            "Consumer complaint narrative": [long_text],
        }
    )
    chunks = chunk_dataframe(df, chunk_size=40, chunk_overlap=10, use_langchain=False)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    # complaint_id and product carried into every chunk
    assert all(c.complaint_id == 101 for c in chunks)
    assert all(c.product == "Credit card" for c in chunks)
    # chunk_index increments from 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_dataframe_to_dict_is_flat():
    df = pd.DataFrame(
        {
            "Complaint ID": [7],
            "Product": ["Personal loan"],
            "Consumer complaint narrative": ["a complaint about fees"],
        }
    )
    chunks = chunk_dataframe(df, use_langchain=False)
    d = chunks[0].to_dict()
    assert d["complaint_id"] == 7
    assert d["product"] == "Personal loan"
    assert "text" in d and "chunk_index" in d


def test_chunk_dataframe_missing_text_column_raises():
    df = pd.DataFrame({"Complaint ID": [1], "Product": ["Credit card"]})
    with pytest.raises(KeyError):
        chunk_dataframe(df)


# --------------------------------------------------------------------------- #
# Helper / summary functions
# --------------------------------------------------------------------------- #

def test_count_chunks():
    chunks = split_text(" ".join(str(i) for i in range(100)), chunk_size=20, chunk_overlap=5)
    assert count_chunks(chunks) == len(chunks)


def test_chunk_length_summary_values():
    summary = chunk_length_summary(["abc", "de", "fghij"])
    assert summary["num_chunks"] == 3
    assert summary["min_chars"] == 2
    assert summary["max_chars"] == 5
    assert summary["total_chars"] == 10


def test_chunk_length_summary_empty():
    summary = chunk_length_summary([])
    assert summary["num_chunks"] == 0
    assert summary["total_chars"] == 0
