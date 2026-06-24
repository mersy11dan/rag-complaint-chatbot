"""Data loading and cleaning utilities for consumer complaint narratives."""

from __future__ import annotations

import re

import pandas as pd

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

# The four product categories CrediTrust focuses on. Matching is done on these
# lowercase keywords because the raw CFPB labels vary (e.g. "Credit card or prepaid card").
TARGET_PRODUCTS = ["credit card", "personal loan", "savings account", "money transfer"]

# Common boilerplate phrases that add no semantic value to a complaint narrative.
# These are stripped before embedding so the model focuses on the actual issue.
BOILERPLATE_PATTERNS = [
    r"i am writing to (?:file|submit|lodge) a complaint",
    r"to whom it may concern",
    r"i am writing to dispute",
    r"this is a complaint (?:about|regarding)",
    r"please be advised that",
    r"thank you for your (?:time|attention|help|assistance)",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw complaints dataset from a CSV file."""
    return pd.read_csv(path, low_memory=False)


def remove_boilerplate(text: str) -> str:
    """Remove common, low-signal boilerplate phrases from a narrative."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate/redactions, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = remove_boilerplate(text)
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)  # remove redaction tokens like "xxxx"
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose narrative is missing, empty, or whitespace-only."""
    narrative = df[NARRATIVE_COL]
    has_narrative = narrative.notna() & (narrative.astype(str).str.strip() != "")
    return df[has_narrative].reset_index(drop=True)


def filter_by_product(df: pd.DataFrame, products: list[str] | None = None) -> pd.DataFrame:
    """Keep only rows whose product matches one of the target product keywords."""
    products = products or TARGET_PRODUCTS
    product_lower = df[PRODUCT_COL].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for keyword in products:
        mask |= product_lower.str.contains(keyword.lower(), na=False)
    return df[mask].reset_index(drop=True)


def filter_complaints(df: pd.DataFrame, products: list[str] | None = None) -> pd.DataFrame:
    """Full filtering step: drop empty narratives, then keep target products."""
    df = filter_empty_narratives(df)
    df = filter_by_product(df, products)
    return df.reset_index(drop=True)
