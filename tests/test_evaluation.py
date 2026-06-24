"""Tests for src.evaluation (uses a fake pipeline; no heavy dependencies)."""

from src.evaluation import (
    EVALUATION_QUESTIONS,
    EvalResult,
    run_evaluation,
    to_markdown_report,
    to_markdown_table,
)


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakePipeline:
    """Returns a deterministic answer + sources for any question."""

    def answer(self, question):
        return {
            "question": question,
            "answer": f"Answer to: {question}",
            "sources": [_FakeChunk("excerpt one | with pipe"), _FakeChunk("excerpt two")],
        }


def test_evaluation_questions_count_in_range():
    assert 5 <= len(EVALUATION_QUESTIONS) <= 10


def test_run_evaluation_returns_one_result_per_question():
    pipeline = _FakePipeline()
    results = run_evaluation(pipeline, questions=["q1", "q2"])
    assert len(results) == 2
    assert all(isinstance(r, EvalResult) for r in results)
    assert results[0].answer == "Answer to: q1"
    assert len(results[0].sources) == 2


def test_markdown_table_has_required_columns_and_escapes_pipes():
    pipeline = _FakePipeline()
    results = run_evaluation(pipeline, questions=["q1"])
    table = to_markdown_table(results)
    for col in ["Question", "Generated Answer", "Retrieved Sources", "Quality Score", "Comments"]:
        assert col in table
    # The pipe inside a source excerpt must be escaped so it doesn't break the table
    assert "\\|" in table


def test_markdown_report_contains_table_and_analysis():
    pipeline = _FakePipeline()
    results = run_evaluation(pipeline, questions=["q1", "q2"])
    report = to_markdown_report(results)
    assert "## Evaluation Table" in report
    assert "## Analysis" in report
    assert "What worked well" in report
