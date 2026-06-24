"""Helper functions shared by the chat UI (app.py)."""

from __future__ import annotations


def format_sources(chunks) -> str:
    """Render retrieved chunks as a human-readable, numbered source list."""
    if not chunks:
        return "_No sources retrieved._"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = getattr(chunk, "text", str(chunk))
        snippet = text[:300] + ("..." if len(text) > 300 else "")
        lines.append(f"**Source {i}:** {snippet}")
    return "\n\n".join(lines)


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to a maximum number of characters with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
