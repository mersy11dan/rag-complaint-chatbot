"""Tests for src.preprocess."""

import pandas as pd

from src.preprocess import clean_text, filter_complaints


def test_clean_text_lowercases_and_strips():
    assert clean_text("  Hello   WORLD  ") == "hello world"


def test_clean_text_removes_redaction_tokens():
    assert "xxxx" not in clean_text("My account XXXX was closed")


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


def test_filter_complaints_drops_missing_narratives():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": ["a complaint", None],
            "Product": ["Credit card", "Credit card"],
        }
    )
    result = filter_complaints(df)
    assert len(result) == 1


def test_filter_complaints_filters_by_product():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": ["x", "y"],
            "Product": ["Credit card", "Mortgage"],
        }
    )
    result = filter_complaints(df, products=["Credit card"])
    assert result["Product"].tolist() == ["Credit card"]
