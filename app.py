"""Interactive chat UI for the RAG complaint chatbot.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.ui_utils import format_sources


def main() -> None:
    st.set_page_config(page_title="CrediTrust Complaint Assistant", page_icon="??")
    st.title("CrediTrust Complaint Assistant")
    st.caption("Ask questions about customer complaints and get grounded answers.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about customer complaints...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # TODO: wire up the RAGPipeline here once the vector store is built.
        answer = "The RAG pipeline is not connected yet."
        sources: list = []

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Sources"):
                st.markdown(format_sources(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
