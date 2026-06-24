"""Tests for src.rag (pure logic; no heavy model dependencies)."""

from src.rag import (
    NO_CONTEXT_MESSAGE,
    Generator,
    RAGPipeline,
    RetrievedChunk,
    build_prompt,
    format_context,
)


def _chunks():
    return [
        RetrievedChunk(text="charged a late fee twice", score=0.9, metadata={"product": "Credit card", "company": "BankX"}),
        RetrievedChunk(text="transfer never arrived", score=0.7, metadata={"product": "Money transfer"}),
    ]


# --------------------------------------------------------------------------- #
# format_context / build_prompt
# --------------------------------------------------------------------------- #

def test_format_context_numbers_sources_and_includes_metadata():
    ctx = format_context(_chunks())
    assert "[Source 1" in ctx and "[Source 2" in ctx
    assert "Credit card" in ctx and "BankX" in ctx
    assert "charged a late fee twice" in ctx


def test_build_prompt_contains_question_and_context_and_rules():
    prompt = build_prompt("Why was I charged a fee?", _chunks())
    assert "Why was I charged a fee?" in prompt
    assert "charged a late fee twice" in prompt
    # The no-context instruction is injected into the rules
    assert NO_CONTEXT_MESSAGE in prompt


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

def test_generator_returns_no_context_message_when_empty():
    gen = Generator(llm=lambda prompt: "should not be called")
    assert gen.generate("anything", []) == NO_CONTEXT_MESSAGE


def test_generator_uses_injected_llm_with_prompt():
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "fake answer"

    gen = Generator(llm=fake_llm)
    answer = gen.generate("Why was I charged a fee?", _chunks())
    assert answer == "fake answer"
    # The LLM received a prompt that embeds the question + context
    assert "Why was I charged a fee?" in captured["prompt"]
    assert "late fee" in captured["prompt"]


# --------------------------------------------------------------------------- #
# RAGPipeline
# --------------------------------------------------------------------------- #

def test_pipeline_combines_retriever_and_generator():
    def fake_retriever(question, top_k):
        assert top_k == 3
        return _chunks()

    gen = Generator(llm=lambda prompt: "grounded answer")
    pipeline = RAGPipeline(fake_retriever, gen, top_k=3)

    result = pipeline.answer("Why was I charged a fee?")
    assert result["question"] == "Why was I charged a fee?"
    assert result["answer"] == "grounded answer"
    assert len(result["sources"]) == 2


def test_pipeline_handles_no_retrieved_chunks():
    pipeline = RAGPipeline(lambda q, k: [], Generator(llm=lambda p: "unused"))
    result = pipeline.answer("totally unrelated question")
    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == []
