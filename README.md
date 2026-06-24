# RAG Complaint Chatbot

An intelligent, evidence-backed complaint-analysis assistant for **CrediTrust Financial**,
built with **Retrieval-Augmented Generation (RAG)**. It turns hundreds of thousands of
unstructured consumer complaint narratives into a queryable knowledge base, so internal
teams can ask plain-English questions and receive grounded, **source-cited** answers.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Solution Overview](#solution-overview)
3. [Project Structure](#project-structure)
4. [EDA & Preprocessing](#eda--preprocessing)
5. [Chunking & Embedding Pipeline](#chunking--embedding-pipeline)
6. [Vector Store Choice](#vector-store-choice)
7. [RAG: Retrieval & Generation](#rag-retrieval--generation)
8. [Evaluation Approach](#evaluation-approach)
9. [Setup](#setup)
10. [Running the UI Locally](#running-the-ui-locally)
11. [Testing](#testing)
12. [Limitations & Next Steps](#limitations--next-steps)

---

## Business Problem

CrediTrust Financial receives a large volume of consumer complaints across several product
lines (credit cards, personal loans, savings accounts, and money transfers). These
complaints arrive as **free-text narratives** — rich with insight but effectively
**unsearchable at scale**. Product managers, compliance, and support teams cannot easily
answer questions like:

- *"What are customers most frustrated about with money transfers?"*
- *"Are there recurring issues with personal loan repayment?"*

Reading complaints manually is slow and unscalable; traditional keyword search misses
paraphrased issues and returns no synthesized answer. The goal of this project is to let a
**non-technical user ask a question in natural language** and get a concise, trustworthy
answer that is **grounded in actual complaint excerpts** — with the sources shown so the
answer can be verified.

---

## Solution Overview

The system implements a classic RAG architecture:

```
Raw complaints  ->  Clean & filter  ->  Chunk  ->  Embed  ->  Vector store
                                                                   |
                  User question  ->  Embed  ->  Retrieve top-k  ---+
                                                       |
                                  Prompt (context + question)  ->  LLM  ->  Grounded answer + sources
```

Each stage is a small, testable module in `src/`, and each analytical stage has a
companion notebook in `notebooks/`.

---

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/
│   └── unittests.yml             # CI: install deps + run pytest on push to main
├── data/
│   ├── raw/                      # Original CFPB complaints (git-ignored, large)
│   └── processed/                # Cleaned & filtered data
├── vector_store/                 # Persisted ChromaDB / FAISS index
├── notebooks/
│   ├── eda_preprocessing.ipynb   # EDA + cleaning/filtering
│   ├── chunking_embedding.ipynb  # Sampling, chunking experiments, indexing
│   └── rag_evaluation.ipynb      # Pipeline evaluation
├── src/
│   ├── preprocess.py             # Loading, cleaning, filtering
│   ├── chunking.py               # Text chunking + metadata
│   ├── embeddings.py             # Embedding model + vector store
│   ├── rag.py                    # Retriever, prompt, generator, pipeline
│   ├── evaluation.py             # Evaluation harness + markdown report
│   └── ui_utils.py               # UI helper functions
├── tests/                        # Unit tests for each module
├── app.py                        # Streamlit chat UI
├── requirements.txt
└── README.md
```

---

## EDA & Preprocessing

**Notebook:** `notebooks/eda_preprocessing.ipynb` · **Module:** `src/preprocess.py`

The exploratory analysis on the full CFPB dataset examines:

- **Complaint distribution by product** — to understand volume and class imbalance.
- **Narrative coverage** — how many complaints actually contain a free-text narrative
  (only those are usable for RAG; the rest are dropped).
- **Narrative length distribution** (in words) — right-skewed, which directly informs the
  chunking strategy.
- **Outliers** — very short (`< 5` words, low signal) and very long (`> 500` words, will
  span multiple chunks) narratives.

Preprocessing (`src/preprocess.py`) then:

- **Filters to the four target products** via keyword matching, robust to verbose CFPB
  labels (e.g. `"Credit card or prepaid card"` → `credit card`).
- **Drops empty/whitespace-only narratives.**
- **Cleans text** (`clean_text`): lowercasing, removal of boilerplate preambles
  (e.g. *"I am writing to file a complaint"*) and `XXXX` redaction tokens, and whitespace
  normalization.

The cleaned, filtered subset is written to `data/processed/filtered_complaints.csv`.

---

## Chunking & Embedding Pipeline

**Notebook:** `notebooks/chunking_embedding.ipynb` · **Modules:** `src/chunking.py`, `src/embeddings.py`

**Sampling.** To iterate quickly, the pipeline draws a **stratified-by-product sample of
~12,000 complaints** (proportional allocation, capped per product, `random_state=42`),
preserving the full dataset's product balance.

**Chunking** (`src/chunking.py`). Long narratives are split into smaller, overlapping
chunks. The splitter prefers LangChain's `RecursiveCharacterTextSplitter` and falls back to
a dependency-free, word-boundary-aware splitter when LangChain is unavailable. After
experimenting with several `(chunk_size, overlap)` combinations, the final choice is:

- **`chunk_size = 500` characters, `chunk_overlap = 50` (10%).**
- Rationale: captures a complete thought without over-fragmenting, keeps the total chunk
  count manageable, preserves continuity across boundaries via overlap, and stays
  comfortably within the embedding model's 256-token window.

Each chunk preserves **traceability metadata**: `complaint_id`, `product_category`,
`product`, `issue`, `company`, `state`, `date_received`, `chunk_index`, and `total_chunks`.

**Embedding** (`src/embeddings.py`). Chunks are embedded with
**`sentence-transformers/all-MiniLM-L6-v2`** (384-dimensional, L2-normalized). This model is
a strong, lightweight default: fast on CPU, good semantic quality, and a small footprint
suitable for a local vector store.

---

## Vector Store Choice

The default backend is **ChromaDB**, with **FAISS** supported as an alternative
(`backend="faiss"`).

**Why ChromaDB:**

- **Native metadata storage** — each vector keeps its full complaint metadata alongside the
  document, which is essential for source attribution and future filtered retrieval
  (e.g. "only credit card complaints").
- **Built-in persistence** — `PersistentClient` writes the collection to `vector_store/`
  with no extra bookkeeping.
- **Simple, batteries-included API** for add/query with cosine similarity.

FAISS is provided for scenarios that need raw speed or a portable index file; in that mode
the index is written to `index.faiss` and metadata to `metadata.json`. Embeddings are
L2-normalized so cosine similarity reduces to an inner product, used consistently by both
backends.

---

## RAG: Retrieval & Generation

**Module:** `src/rag.py`

- **Retriever.** Embeds the user's question with the **same** `all-MiniLM-L6-v2` model used
  at indexing time, queries the vector store, and returns the **top-k** chunks (default
  `k=5`) as `RetrievedChunk` objects (text + similarity score + metadata).
- **Prompt template.** Instructs the LLM to answer **only** from the retrieved context, to
  stay concise and factual, and — critically — to reply with an explicit *"I don't have
  enough information…"* message when the context is insufficient. This is the main guardrail
  against hallucination.
- **Generator.** Combines the prompt + retrieved chunks + question and calls the LLM
  (default `google/flan-t5-base`). If retrieval returns nothing, it short-circuits to the
  safe "not enough information" response without calling the model.
- **Pipeline.** `RAGPipeline.answer(question)` runs retrieve → generate and returns
  `{question, answer, sources}`. The `build_rag_pipeline()` factory assembles everything
  from a persisted store.

The embedder, vector store, and LLM are all **injectable**, keeping the core logic clean and
unit-testable without heavy model downloads.

---

## Evaluation Approach

**Notebook:** `notebooks/rag_evaluation.ipynb` · **Module:** `src/evaluation.py`

Evaluation is **qualitative and reproducible**:

- A set of **representative questions** spans all four products, plus a deliberately
  **out-of-scope control** question to confirm the system refuses to answer when it should.
- Each question is run through the pipeline; the answer and top retrieved sources are
  collected.
- Results are rendered as a **markdown evaluation table** with columns: *Question, Generated
  Answer, Retrieved Sources, Quality Score (1–5), Comments*. A human reviewer fills in the
  score and comments.
- The notebook also produces a written **strengths / weaknesses** analysis, and the full
  report can be saved to `notebooks/evaluation_report.md` for direct inclusion in the final
  deliverable.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) download NLP models used during preprocessing
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

**Build the knowledge base** before running the UI:

1. Place the raw CFPB complaints CSV in `data/raw/`.
2. Run `notebooks/eda_preprocessing.ipynb` to produce `data/processed/filtered_complaints.csv`.
3. Run `notebooks/chunking_embedding.ipynb` to chunk, embed, and persist the vector store to
   `vector_store/`.

---

## Running the UI Locally

The app is a **Streamlit** chat interface wired to `src/rag.py`:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (default `http://localhost:8501`). The UI
provides:

- A **text box** for the question and an **Ask** button.
- The **AI-generated answer**, grounded in retrieved complaints.
- The **retrieved source excerpts** below the answer, each with its product/company/state
  metadata and similarity score.
- A **Clear** button and clickable **example questions**.

The pipeline (embedding + LLM models) is cached after first load. The first run downloads
the models, so it may take a few minutes.

---

## Testing

Unit tests cover preprocessing, chunking, the RAG logic, and the evaluation harness. Heavy
dependencies are mocked/injected so tests run fast and without model downloads.

```bash
pytest tests/ -v
```

CI runs the suite automatically on every push and pull request to `main`
(`.github/workflows/unittests.yml`).

---

## Limitations & Next Steps

**Current limitations**

- **Extractive, not aggregative.** The model summarizes a handful of retrieved excerpts
  rather than reasoning over the full complaint volume, so it cannot reliably produce exact
  counts or trends.
- **Retrieval precision.** Some top-k chunks can be only loosely relevant; relevance depends
  on chunk size and the embedding model.
- **Lightweight LLM.** The default `flan-t5-base` favors speed over fluency; complex
  questions may receive terse answers.
- **Sampled index.** The shipped pipeline indexes a ~12k stratified sample for iteration
  speed, not the entire dataset.
- **Product imbalance.** Broad questions may over-represent the most common product.

**Next steps**

- Add a **cross-encoder re-ranker** over the top-k results to sharpen relevance.
- Support **metadata-filtered retrieval** (e.g. by product, date range, or company).
- Scale indexing to the **full dataset** with batched, incremental embedding.
- Swap in a **stronger LLM** and add response streaming in the UI.
- Add **quantitative retrieval metrics** (e.g. hit-rate / MRR) alongside the qualitative review.
