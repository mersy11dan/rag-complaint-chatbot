# RAG Complaint Chatbot — Week 7

An intelligent, evidence-backed complaint-analysis assistant for **CrediTrust Financial**,
built with Retrieval-Augmented Generation (RAG). It turns unstructured consumer complaint
narratives into actionable insights so internal teams can ask plain-English questions and
receive grounded, source-cited answers.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Preprocessing](#preprocessing)
5. [Chunking](#chunking)
6. [Embeddings](#embeddings)
7. [RAG Pipeline](#rag-pipeline)
8. [User Interface](#user-interface)
9. [Evaluation](#evaluation)
10. [Report](#report)

---

## Overview

> _Describe the business problem, the dataset (CFPB consumer complaints), and the goal of
> the project: enabling non-technical teams to query complaint narratives in natural language._

- **Problem:** ...
- **Dataset:** ...
- **Objective:** ...

---

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/
│   └── unittests.yml        # CI: install deps + run pytest on push to main
├── data/
│   ├── raw/                 # Original, immutable datasets
│   └── processed/           # Cleaned & filtered data
├── vector_store/            # Persisted FAISS / Chroma index
├── notebooks/               # EDA & experimentation
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data loading & cleaning
│   ├── chunking.py          # Text chunking
│   ├── embeddings.py        # Embedding & vector store
│   ├── rag.py               # RAG pipeline (retrieve + generate)
│   ├── evaluation.py        # Pipeline evaluation
│   └── ui_utils.py          # UI helper functions
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   └── test_chunking.py
├── app.py                   # Streamlit / Gradio chat UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Optional: download NLP models
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

---

## Preprocessing

> _Document the data loading, cleaning, and filtering steps (`src/preprocess.py`)._

- Load raw complaints from `data/raw/`.
- Filter to relevant product categories.
- Clean narratives (lowercasing, redaction-token removal, whitespace normalization).
- Save cleaned data to `data/processed/`.

---

## Chunking

> _Explain the chunking strategy (`src/chunking.py`): chunk size, overlap, and rationale._

- Strategy: word/character-based splitting with overlap.
- Chosen `chunk_size` and `overlap`: ...
- Why these values: ...

---

## Embeddings

> _Describe the embedding model and vector store (`src/embeddings.py`)._

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (or alternative).
- Vector store: FAISS / ChromaDB.
- Index persisted to `vector_store/`.

---

## RAG Pipeline

> _Detail retrieval + generation (`src/rag.py`)._

- Retriever: top-k semantic search over the vector store.
- Prompt template: instructs the LLM to answer only from retrieved context.
- Generator: LLM used for answer synthesis.

---

## User Interface

> _Describe the chat UI (`app.py`)._

```bash
streamlit run app.py
# or, if using Gradio:
# python app.py
```

- Chat input, streamed responses, and displayed source excerpts.

---

## Evaluation

> _Summarize qualitative/quantitative evaluation (`src/evaluation.py`)._

| Question | Generated Answer | Retrieved Sources | Score (1-5) | Comments |
|----------|------------------|-------------------|-------------|----------|
| ...      | ...              | ...               | ...         | ...      |

---

## Report

> _Link to or embed the final report: methodology, results, limitations, and next steps._

- Key findings: ...
- Limitations: ...
- Future work: ...

---

## Testing

```bash
pytest tests/ -v
```
