"""Data loading and cleaning utilities for consumer complaint narratives."""

from __future__ import annotations

import re

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw complaints dataset from a CSV file."""
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate, and normalize whitespace in a narrative."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)  # remove redaction tokens like "xxxx"
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_complaints(df: pd.DataFrame, products: list[str] | None = None) -> pd.DataFrame:
    """Keep only rows with a non-empty narrative and (optionally) selected products."""
    df = df.dropna(subset=["Consumer complaint narrative"])
    if products:
        df = df[df["Product"].isin(products)]
    return df.reset_index(drop=True)
