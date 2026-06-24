"""Helper functions shared by the chat UI (app.py)."""

from __future__ import annotations


def source_label(chunk) -> str:
    """Build a short metadata label like 'Credit card · BankX · CA' for a chunk."""
    meta = getattr(chunk, "metadata", {}) or {}
    parts = [
        meta.get("product") or meta.get("product_category"),
        meta.get("company"),
        meta.get("state"),
    ]
    return " · ".join(str(p) for p in parts if p)


def format_sources(chunks, max_chars: int = 300) -> str:
    """Render retrieved chunks as a human-readable, numbered markdown source list."""
    if not chunks:
        return "_No sources retrieved._"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = getattr(chunk, "text", str(chunk))
        snippet = text[:max_chars] + ("..." if len(text) > max_chars else "")
        label = source_label(chunk)
        header = f"**Source {i}**" + (f" — _{label}_" if label else "")
        lines.append(f"{header}\n\n{snippet}")
    return "\n\n".join(lines)


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to a maximum number of characters with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
