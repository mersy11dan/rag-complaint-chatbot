"""Evaluation utilities for the complaint RAG system.

Runs a set of representative questions through the RAG pipeline and renders a
markdown evaluation table (plus a short analysis) that can be pasted directly
into the final report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 8 representative questions spanning the four target products plus a
# deliberately out-of-scope question to test the "insufficient context" path.
EVALUATION_QUESTIONS = [
    "What are the most common credit card complaints?",
    "Why do customers dispute credit card late fees?",
    "What problems do customers report about personal loans?",
    "Are customers having issues with personal loan repayment or interest charges?",
    "What complaints do people have about savings accounts?",
    "Why are customers unhappy with money transfers?",
    "Do customers report delayed or failed money transfers?",
    "What is the weather forecast for tomorrow?",  # out-of-scope control
]

# Number of retrieved source excerpts to show in the table.
SOURCES_IN_TABLE = 2
# Max characters per source excerpt / answer cell (keeps the table readable).
SOURCE_PREVIEW_CHARS = 200
ANSWER_PREVIEW_CHARS = 400


@dataclass
class EvalResult:
    """One evaluated question with its answer, sources, and review fields."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    quality_score: int | str = ""  # 1-5, filled in by a human reviewer
    comments: str = ""


# --------------------------------------------------------------------------- #
# Running the pipeline
# --------------------------------------------------------------------------- #

def run_evaluation(
    pipeline: Any,
    questions: list[str] | None = None,
    n_sources: int = SOURCES_IN_TABLE,
) -> list[EvalResult]:
    """Run the RAG pipeline over each question and collect the results."""
    questions = questions or EVALUATION_QUESTIONS
    results: list[EvalResult] = []
    for question in questions:
        out = pipeline.answer(question)
        sources = [getattr(s, "text", str(s)) for s in out.get("sources", [])[:n_sources]]
        results.append(
            EvalResult(question=question, answer=out.get("answer", ""), sources=sources)
        )
    return results


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def _clean_cell(text: str, max_chars: int) -> str:
    """Make a string safe and compact for a single markdown table cell."""
    text = " ".join(str(text).split())  # collapse newlines/whitespace
    text = text.replace("|", "\\|")  # escape pipes so they don't break the table
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _format_sources(sources: list[str]) -> str:
    if not sources:
        return "_No sources retrieved_"
    parts = [f"{i}. {_clean_cell(s, SOURCE_PREVIEW_CHARS)}" for i, s in enumerate(sources, 1)]
    return "<br>".join(parts)


def to_markdown_table(results: list[EvalResult]) -> str:
    """Render evaluation results as a markdown table."""
    header = (
        "| # | Question | Generated Answer | Retrieved Sources | Quality Score | Comments |\n"
        "|---|----------|------------------|-------------------|:-------------:|----------|"
    )
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            "| {n} | {q} | {a} | {s} | {score} | {c} |".format(
                n=i,
                q=_clean_cell(r.question, 200),
                a=_clean_cell(r.answer, ANSWER_PREVIEW_CHARS),
                s=_format_sources(r.sources),
                score=r.quality_score if r.quality_score != "" else " ",
                c=_clean_cell(r.comments, 200) if r.comments else " ",
            )
        )
    return "\n".join([header, *rows])


DEFAULT_ANALYSIS = """\
### What worked well
- _(Fill in: e.g., retrieval surfaced on-topic complaint excerpts for in-scope product questions.)_
- _(e.g., The model correctly declined to answer the out-of-scope control question.)_

### What could be improved
- _(Fill in: e.g., answers were occasionally too generic; tighten the prompt or increase top-k.)_
- _(e.g., Some retrieved chunks were only loosely relevant; consider re-ranking or smaller chunks.)_

### Notes
- Quality scores are on a 1-5 scale (1 = poor, 5 = excellent), assigned by a human reviewer.
"""


def to_markdown_report(
    results: list[EvalResult],
    analysis: str = DEFAULT_ANALYSIS,
    title: str = "RAG Evaluation",
) -> str:
    """Assemble a full, paste-ready markdown report (table + analysis)."""
    return (
        f"# {title}\n\n"
        f"Evaluated **{len(results)}** representative questions against the RAG pipeline.\n\n"
        "## Evaluation Table\n\n"
        f"{to_markdown_table(results)}\n\n"
        "## Analysis\n\n"
        f"{analysis}\n"
    )


def save_markdown_report(
    results: list[EvalResult],
    path: str = "notebooks/evaluation_report.md",
    analysis: str = DEFAULT_ANALYSIS,
    title: str = "RAG Evaluation",
) -> str:
    """Write the markdown report to ``path`` and return the report text."""
    report = to_markdown_report(results, analysis=analysis, title=title)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report


def evaluate(
    pipeline: Any,
    questions: list[str] | None = None,
    path: str | None = "notebooks/evaluation_report.md",
) -> tuple[list[EvalResult], str]:
    """Run the evaluation end-to-end: execute the pipeline and render the report.

    Returns the list of results and the markdown report. If ``path`` is given,
    the report is also written to disk.
    """
    results = run_evaluation(pipeline, questions)
    if path:
        report = save_markdown_report(results, path=path)
    else:
        report = to_markdown_report(results)
    return results, report
