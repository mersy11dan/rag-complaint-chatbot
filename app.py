"""Interactive complaint chatbot UI (Streamlit).

Wires the Streamlit front-end to the RAG pipeline in ``src/rag.py``. The user
types a question, clicks **Ask**, and sees the AI-generated answer plus the
retrieved complaint excerpts that grounded it. A **Clear** button resets the view.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.rag import build_rag_pipeline
from src.ui_utils import source_label

PERSIST_DIR = "vector_store"
BACKEND = "chroma"  # or "faiss"
TOP_K = 5

EXAMPLE_QUESTIONS = [
    "What are the most common credit card complaints?",
    "Why are customers unhappy with money transfers?",
    "What problems do customers report about personal loans?",
    "What complaints do people have about savings accounts?",
]


@st.cache_resource(show_spinner="Loading the RAG pipeline (first run downloads models)...")
def load_pipeline():
    """Build the RAG pipeline once and cache it across reruns."""
    return build_rag_pipeline(backend=BACKEND, persist_dir=PERSIST_DIR, top_k=TOP_K)


def init_state() -> None:
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("sources", [])


def clear() -> None:
    st.session_state.question = ""
    st.session_state.answer = None
    st.session_state.sources = []


def run_query(question: str) -> None:
    """Run the pipeline and store the answer + sources in session state."""
    pipeline = load_pipeline()
    with st.spinner("Searching complaints and generating an answer..."):
        result = pipeline.answer(question)
    st.session_state.answer = result["answer"]
    st.session_state.sources = result["sources"]


def main() -> None:
    st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="centered")
    init_state()

    st.title("CrediTrust Complaint Assistant")
    st.caption(
        "Ask a question about customer complaints. Answers are grounded in real complaint "
        "narratives retrieved from the knowledge base."
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "- Retrieval-Augmented Generation over consumer complaints.\n"
            "- Answers cite the complaint excerpts used.\n"
            "- If the knowledge base lacks the answer, the assistant says so."
        )
        st.subheader("Try an example")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.question = q

    question = st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. Why are customers unhappy with money transfers?",
    )

    col_ask, col_clear, _ = st.columns([1, 1, 4])
    ask_clicked = col_ask.button("Ask", type="primary", use_container_width=True)
    col_clear.button("Clear", on_click=clear, use_container_width=True)

    if ask_clicked:
        if question.strip():
            run_query(question.strip())
        else:
            st.warning("Please enter a question first.")

    if st.session_state.answer is not None:
        st.markdown("### Answer")
        st.write(st.session_state.answer)

        st.markdown("### Retrieved sources")
        sources = st.session_state.sources
        if not sources:
            st.info("No source excerpts were retrieved for this question.")
        else:
            for i, chunk in enumerate(sources, start=1):
                label = source_label(chunk)
                score = getattr(chunk, "score", None)
                title = f"Source {i}" + (f" — {label}" if label else "")
                if score is not None:
                    title += f"  (similarity: {score:.3f})"
                with st.expander(title, expanded=(i == 1)):
                    st.write(getattr(chunk, "text", str(chunk)))


if __name__ == "__main__":
    main()
