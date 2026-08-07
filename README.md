# Clinical RAG Assistant

A RAG-based (Retrieval-Augmented Generation) Clinical Decision Support System that answers medical questions using real documents — not guesses. Every answer comes with a confidence score, sentence-level verification, and evidence showing exactly which document and page it came from.

---

## Why We're Building This

Doctors and students deal with huge amounts of medical documents (research papers, guidelines, patient notes). Two big problems exist today:
1. **Normal search is dumb** — keyword search doesn't understand meaning, so it often misses relevant info.
2. **Normal AI chatbots hallucinate** — they can confidently give wrong medical information, which is dangerous in healthcare.

**Our solution:** A system that only answers using the documents you give it, tells you *how confident* it is in each answer, and shows you exactly where that answer came from — so nothing is a blind guess.

---

## What The Final Outcome Will Be

A web app where a user can:
- Upload any medical PDF
- Ask questions in plain English
- Get an answer that is grounded in that PDF (not made up)
- See a confidence score (High/Medium/Low) for every answer
- See exactly which PDF and page number the answer came from
- See a sentence-by-sentence breakdown of how well-supported each part of the answer is
- Have every question, answer, and confidence score saved for later review

This turns AI from "maybe right, maybe not" into a transparent, traceable, trustworthy clinical assistant.

---

## The Big Picture — How It Works (Simple Version)

```
PDF Upload → Split into small pieces (with page numbers tracked) → Convert to "meaning numbers" → Store
Question → Convert to "meaning numbers" → FAISS finds 20 possible matches → Cross-Encoder reranks them → Best 5 selected
Best 5 chunks + Question → Sent to AI model → Generates answer
Answer → Split into sentences → Each sentence checked against source → Confidence + Evidence produced → Saved to database
```

That's the whole idea. Everything below is how we actually built it, step by step.

---

## Models Used

| Purpose | Model |
|---------|-------|
| Embeddings (text → vectors) | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Reranking retrieved chunks | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformers) |
| Answer generation | `llama-3.3-70b-versatile` (Meta's Llama 3, hosted via Groq API) |

We did not train any model ourselves — both are pre-trained, free, and publicly available. We built the *system* (retrieval, reranking, verification, confidence scoring) around them, which is the core idea of RAG.

---

## Tech Stack (What We Used and Why)

| Tool | Purpose | Why we chose it |
|------|---------|------------------|
| Python | Main programming language | Best support for AI/ML tools |
| LangChain | Splits documents into chunks | Industry standard, easy to use |
| SentenceTransformers | Converts text into vectors (meaning) + reranking | Free, fast, runs on your own PC |
| FAISS | Stores and searches vectors | Free, no server setup needed |
| pypdf | Reads text from uploaded PDFs | Free, simple PDF text extraction |
| Llama 3 (via Groq API) | Generates the actual answer | Free tier, very fast responses |
| FastAPI | Turns our code into a web API | Simple, fast, auto-generates docs |
| python-multipart | Lets FastAPI accept file uploads | Required for the `/upload` endpoint |
| scikit-learn | Checks if answer matches source | Simple way to catch AI hallucination |
| PostgreSQL + psycopg2 | Stores every question/answer permanently | Free, reliable, industry-standard database |
| matplotlib | Generates evaluation graphs | Free, standard Python plotting library |
| requests | Used by our evaluation script to call our own API | Simple way to automate testing |

---

## Step-by-Step: What We Did (In Order)

### Step 1 — Environment Setup
Installed Python, created a project folder, and set up a virtual environment (an isolated space for this project's tools so they don't conflict with other projects).

```bash
python -m venv venv
venv\Scripts\activate
pip install langchain sentence-transformers faiss-cpu
```

### Step 2 — Document Ingestion (`ingest.py`)
This is where we prepare a document to be searchable:
- **Chunking**: Split the document text into small pieces (`chunk_size=300-500`). Small pieces work better than giant blocks of text, but too small breaks sentences apart, so this was tuned after testing.
- **Metadata tracking**: While chunking, we track which PDF and which page each chunk came from, page by page — not just raw text.
- **Embedding**: Convert each chunk's text into a vector — a list of numbers that represents its *meaning* (not just the words).
- **Storing**: Save all these vectors into FAISS, a database built specifically for fast vector searching. The metadata (PDF name, page number, chunk ID, text) is stored alongside in `chunks.pkl`.

Think of it like: turning a book into a searchable index, but the index understands meaning, not just keywords, and remembers exactly which page each piece of information came from.

**Note:** `ingest.py` is only meant to be run ONCE, at the very beginning, to build the base vector store. After that, new documents are added through the `/upload` API endpoint instead — running `ingest.py` again will wipe and rebuild everything from scratch.

### Step 3 — Retriever + Cross-Encoder Reranking
This is how we find the right information for a question, in two stages:
- **Stage 1 (FAISS)**: The question is converted into a vector, and FAISS quickly finds the top 20 chunks that are roughly closest in meaning. This is fast but approximate.
- **Stage 2 (Cross-Encoder reranking)**: Each of those 20 candidate chunks is compared directly against the question using a cross-encoder model, which reads the question and chunk together and gives a much more precise relevance score. The top 5 after reranking are kept.

This two-stage "retrieve-then-rerank" approach is the same pattern used in real production RAG systems — FAISS narrows things down fast, then the cross-encoder picks the genuinely best matches from that shortlist.

### Step 4 — Answer Generation (`generator.py`)
This is where the AI actually writes an answer:
- We take the top 5 reranked chunks + the user's question and build an instruction ("prompt") that says: *"Answer using the same key terms and phrasing as the context, and follow any length/format instructions in the question."*
- This prompt is sent to Llama 3 (a large language model) hosted on Groq's free servers.
- Llama 3 sends back a generated answer based only on what we gave it.

### Step 5 — Combining Everything (`main.py`)
This file merges ingestion + retrieval + generation into one live flow that takes a real question from the user and runs it through the whole process automatically.

### Step 6 — Sentence-Level Verification
This is our detailed safety check against hallucination:
- The generated **answer** is split into individual sentences.
- The retrieved **source chunks** are also split into individual sentences.
- Each answer sentence is compared (using cosine similarity) against every source sentence, and matched to whichever one supports it best.
- Each sentence gets a status: **Supported**, **Partially Supported**, or **Unsupported** — giving a granular, sentence-by-sentence trust breakdown instead of one vague number for the whole answer.

### Step 7 — Confidence Engine
Instead of relying on a single similarity score, the Confidence Engine combines multiple signals into one honest final judgment:
- Overall retrieval quality (how relevant were the retrieved chunks to begin with)
- The sentence verification results (how many sentences were Supported vs Unsupported)
- **Strict rule**: if even ONE sentence in the answer is "Unsupported," the whole answer is marked **Low confidence** — even if other sentences scored well. This is intentionally strict, since the goal is catching hallucination, not maximizing the score.

Final labels shown to the user: 🟢 High, 🟡 Medium, 🔴 Low.

### Step 8 — Evidence Extraction
For every answer, we identify the single best-supporting sentence and trace it back to its exact source:
- **Source PDF name**
- **Page number**
- **The exact supporting sentence**

This is what allows the (future) frontend to show a "View Evidence" button next to every answer.

### Step 9 — FastAPI Backend (`api.py`)
We wrapped the entire pipeline into a web API so it can be accessed over the internet (not just from a terminal). This is what our future React frontend will talk to.

```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload
```

Test it at: `http://127.0.0.1:8000/docs`

### Step 10 — PDF Upload Support (`/upload` endpoint in `api.py`)
Instead of manually editing a `.txt` file, users can upload any medical PDF directly through the API:
- The PDF is saved into an `uploads/` folder
- Text is extracted page by page using `pypdf`, tracking page numbers
- The text is chunked, embedded, and added to the existing FAISS store (without deleting what's already there)
- A duplicate-check prevents the same file from being processed twice
- We've tested this with multiple real PDFs (diabetes, hypertension, asthma, and Primary Health Care fact sheets) covering different topics in the same knowledge base

### Step 11 — PostgreSQL Integration
Every question, answer, confidence level, and score is saved permanently to a PostgreSQL database (`clinical_rag`), in a table called `chat_history`. This means:
- Nothing is lost when the server restarts
- We can later build analytics or a history view showing past questions and answers
- Database credentials are kept in `.env`, never hardcoded in the code

### Step 12 — Evaluation Module
To measure real performance instead of just eyeballing a few test questions, we built an automated evaluation pipeline:
- `test_questions.py` holds a list of test questions covering every uploaded PDF
- `evaluate.py` automatically sends each question to our own API, and records the confidence, score, response time, and cited source PDF for every answer
- `analyze_results.py` summarizes this into overall statistics (e.g., % High/Medium/Low confidence, average score, average response time) and generates two graphs:
  - Confidence Level Distribution (bar chart)
  - Score per Question (bar chart)

This gives us real, honest, quantitative proof of how well the system performs — including where it's weaker — instead of just claiming it works.

---

## Current Project Status

| Component | Status |
|-----------|--------|
| Document ingestion (chunking + embeddings) | Done |
| Metadata storage (PDF name, page number, chunk ID) | Done |
| Vector storage & search (FAISS) | Done |
| Cross-encoder reranking | Done |
| Answer generation (Llama 3 via Groq) | Done |
| Sentence-level verification | Done |
| Confidence Engine (combined scoring) | Done |
| Evidence extraction (source PDF + page) | Done |
| FastAPI backend | Done |
| PDF upload support (multiple documents) | Done |
| PostgreSQL (chat history) | Done |
| Evaluation module with graphs | Done |
| React frontend | Not started |
| OCR for images/diagrams in PDFs | Not started (evaluated, decided to skip for now) |
| Feedback system (thumbs up/down) | Not started |

---

## Important Notes

- **We didn't train any AI model.** We use pre-trained models (SentenceTransformers, cross-encoder, Llama 3) and built a smart *system* around them — that's the whole idea of RAG (Retrieval-Augmented Generation).
- **API keys and DB passwords must never be committed to GitHub.** We use a `.env` file (kept out of GitHub via `.gitignore`) to store the Groq API key and database password securely.
- **Never run `ingest.py` after uploading documents via the API** — it rebuilds the vector store from scratch and wipes uploaded content. Use `/upload` for adding documents after the initial setup.
- Confidence scores of 55-75% on well-answered, correctly grounded questions are normal and expected — a RAG system that scores 95%+ on every answer is usually just copy-pasting text instead of genuinely generating a helpful response.
- OCR (extracting text from diagrams/images inside PDFs) was considered but intentionally not implemented yet — it requires external software installs (Tesseract, Poppler) with real risk of setup issues, and we hadn't confirmed our current PDFs actually needed it.
- Everything used so far is **completely free** — no paid tools or subscriptions. Groq's free tier allows 1,000 requests/day, more than enough for development and demos.

---

## Installation History (In the Order We Actually Installed Them)

```bash
pip install langchain sentence-transformers faiss-cpu --timeout 100
```
Core RAG tools — LangChain for chunking, SentenceTransformers for embeddings, FAISS for the vector database.

```bash
pip install langchain-text-splitters
```
Newer LangChain versions moved the text splitter into its own package — needed for `RecursiveCharacterTextSplitter`.

```bash
pip install groq
```
Groq's SDK — lets our code call the Llama 3 model over their API.

```bash
pip install python-dotenv
```
Loads secret values (API keys, DB password) from a `.env` file instead of hardcoding them in the code.

```bash
pip install scikit-learn
```
Provides `cosine_similarity`, used in our validation layer to check if an answer matches the source.

```bash
pip install fastapi uvicorn
```
FastAPI builds the web API; uvicorn is the server that actually runs it.

```bash
pip install pypdf
```
Reads and extracts text from uploaded PDF files.

```bash
pip install huggingface_hub
```
Used to pre-download the embedding model directly, to avoid interrupted downloads during normal runs.

```bash
pip install psycopg2-binary
```
Lets Python connect to and run queries against our PostgreSQL database.

```bash
pip install python-multipart
```
Required by FastAPI to handle file uploads (needed for the `/upload` endpoint).

```bash
pip install requests
```
Used by our evaluation script to automatically call our own API with test questions.

```bash
pip install matplotlib
```
Used to generate the confidence distribution and score graphs for our evaluation report.

Note: `sentence-transformers` (already installed) also provides the `CrossEncoder` class used for reranking — no separate install was needed for that.

---

## Quick Setup Guide (For Teammates)

1. Clone the repo
2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install langchain langchain-text-splitters sentence-transformers faiss-cpu groq python-dotenv scikit-learn fastapi uvicorn pypdf psycopg2-binary python-multipart requests matplotlib
```
4. Create your own `.env` file with:
```
GROQ_API_KEY=your_own_key_here
DATABASE_PASSWORD=your_own_db_password_here
```
5. Install PostgreSQL, then create the database and table:
```sql
CREATE DATABASE clinical_rag;

-- (connect to clinical_rag database, then run:)
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT,
    confidence TEXT,
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```
6. Run the ingestion once to build the base vector store:
```bash
python ingest.py
```
7. Run it as an API:
```bash
uvicorn api:app --reload
```
8. Test at `http://127.0.0.1:8000/docs`:
- Use `POST /upload` to add more PDF documents
- Use `POST /ask` to ask questions

9. To evaluate performance:
```bash
python evaluate.py
python analyze_results.py
```
Check `evaluation_results.json`, `confidence_distribution.png`, and `score_per_question.png` for results.
