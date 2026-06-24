"""Text-cleaning and filtering test suite for src.preprocess."""

import pandas as pd

from src.preprocess import (
    TARGET_PRODUCTS,
    clean_text,
    filter_by_product,
    filter_complaints,
    filter_empty_narratives,
    remove_boilerplate,
)


# --------------------------------------------------------------------------- #
# 1. Boilerplate removal
# --------------------------------------------------------------------------- #

def test_remove_boilerplate_strips_known_phrase():
    text = "To whom it may concern, my card was charged twice."
    result = remove_boilerplate(text)
    assert "to whom it may concern" not in result.lower()
    assert "charged twice" in result.lower()


def test_clean_text_removes_boilerplate_preamble():
    text = "I am writing to file a complaint about a late fee."
    result = clean_text(text)
    assert "writing to file a complaint" not in result
    assert "late fee" in result


def test_clean_text_removes_redaction_tokens():
    assert "xxxx" not in clean_text("My account XXXX was closed")


# --------------------------------------------------------------------------- #
# 2. Lowercase normalization
# --------------------------------------------------------------------------- #

def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_normalizes_whitespace_and_case():
    assert clean_text("  Multiple   SPACES Here  ") == "multiple spaces here"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


# --------------------------------------------------------------------------- #
# 3. Empty narrative filtering
# --------------------------------------------------------------------------- #

def test_filter_empty_narratives_drops_missing_and_blank():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": ["a real complaint", None, "", "   "],
            "Product": ["Credit card"] * 4,
        }
    )
    result = filter_empty_narratives(df)
    assert len(result) == 1
    assert result.loc[0, "Consumer complaint narrative"] == "a real complaint"


# --------------------------------------------------------------------------- #
# 4. Product filtering — keep only the four target products
# --------------------------------------------------------------------------- #

def test_filter_by_product_keeps_only_targets():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": ["x"] * 6,
            "Product": [
                "Credit card or prepaid card",
                "Personal loan",
                "Savings account",
                "Money transfer, virtual currency, or money service",
                "Mortgage",
                "Vehicle loan or lease",
            ],
        }
    )
    result = filter_by_product(df)
    # The two non-target products (Mortgage, Vehicle loan) must be dropped
    assert len(result) == 4
    products = result["Product"].str.lower()
    assert not products.str.contains("mortgage").any()
    assert not products.str.contains("vehicle").any()


def test_filter_complaints_combines_narrative_and_product_filters():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": ["good", None, "also good", "x"],
            "Product": ["Credit card", "Credit card", "Mortgage", "Personal loan"],
        }
    )
    result = filter_complaints(df)
    # Row 1 dropped (no narrative), row 2 dropped (Mortgage) -> 2 remain
    assert len(result) == 2
    assert set(result["Consumer complaint narrative"]) == {"good", "x"}


def test_target_products_has_four_entries():
    assert len(TARGET_PRODUCTS) == 4
