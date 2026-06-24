"""Tests for src.chunking."""

import pytest

from src.chunking import chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("one two three", chunk_size=10, overlap=2)
    assert chunks == ["one two three"]


def test_chunk_text_splits_with_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert len(chunks) > 1
    # consecutive chunks should share the overlapping word
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-1] == second_words[0]


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=2, overlap=2)
