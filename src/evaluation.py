"""Qualitative and quantitative evaluation utilities for the RAG system."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_QUESTIONS = [
    "Why are people unhappy with Buy Now, Pay Later?",
    "What are the most common credit card complaints?",
    "Are there issues with money transfers?",
    "What problems do customers report about personal loans?",
    "How do customers describe their savings account problems?",
]


@dataclass
class EvalResult:
    question: str
    answer: str
    sources: list[str]
    quality_score: int  # 1-5, assigned by a human reviewer
    comments: str = ""


def evaluate(pipeline, questions: list[str] | None = None) -> list[EvalResult]:
    """Run the pipeline over a set of questions for manual scoring."""
    questions = questions or SAMPLE_QUESTIONS
    results: list[EvalResult] = []
    for question in questions:
        out = pipeline.answer(question)
        results.append(
            EvalResult(
                question=question,
                answer=out["answer"],
                sources=[c.text for c in out["sources"][:2]],
                quality_score=0,
            )
        )
    return results
