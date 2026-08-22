# Clinical RAG Assistant

A RAG-based (Retrieval-Augmented Generation) Clinical Decision Support System that answers medical questions using real, uploaded documents — not guesses. Every answer comes with a confidence score, sentence-level verification, and evidence showing exactly which document and page it came from.

---

## Why We're Building This

Doctors and students deal with huge amounts of medical documents (research papers, guidelines, patient notes). Two big problems exist today:

1. **Normal search is dumb** — keyword search doesn't understand meaning, so it often misses relevant info.
2. **Normal AI chatbots hallucinate** — they can confidently give wrong medical information, which is dangerous in healthcare.

**Our solution:** A system that only answers using the documents you give it, tells you *how confident* it is in each answer, shows you exactly where that answer came from, and honestly says "I don't know" when it isn't sure — instead of guessing.

---

## What The System Actually Does

A user can:
- Upload any medical PDF
- Ask questions in plain English
- Get an answer grounded in that PDF (not made up)
- See a confidence score (High / Medium / Low) for every answer
- See exactly which PDF and page number the answer came from
- See a sentence-by-sentence breakdown of how well-supported each part of the answer is
- Get an honest "I don't have enough information" message instead of a shaky guess, when confidence is too low
- Give feedback (👍 / 👎) on any answer
- Choose between **Strict mode** (answers only from documents) or **Direct mode** (plain LLM, no restriction) — useful for comparing the two
- Choose which AI model generates the answer
- Have every question, answer, confidence score, and response time saved for later review
- Access everything only with a valid API key (the system is secured, not open to anyone)

---

## The Big Picture — How It Works

```
PDF Upload → Extract text (page by page) → Chunk → Embed → Store in FAISS + BM25 index

Question → Embed → FAISS (semantic, top 20) + BM25 (keyword, top 20) → merge
         → Cross-Encoder reranks combined candidates → Best 5 chunks selected

Best 5 chunks + Question → Sent to LLM → Generates answer

Answer → Split into sentences → Each checked against source → Confidence + Evidence produced
       → If confidence is too low → replaced with an honest "I don't know" message
       → Saved to database, ready for user feedback
```

That's the full idea. Everything below explains how each part was actually built, in plain language.

---

## Models Used

| Purpose | Model |
|---------|-------|
| Embeddings (text → vectors) | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Reranking retrieved chunks | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformers) |
| Keyword search | BM25 (`rank_bm25`) |
| Answer generation | `openai/gpt-oss-120b` (default, via Groq API) — user can switch to other Groq-hosted models |

No model was trained by us — all are pre-trained, free, and publicly available. We built the *system* (retrieval, reranking, verification, confidence scoring, security) around them — that's the core idea of RAG.

---

## Tech Stack (What We Used and Why)

| Tool | Purpose | Why we chose it |
|------|---------|------------------|
| Python | Main programming language | Best support for AI/ML tools |
| LangChain | Splits documents into chunks | Industry standard, easy to use |
| SentenceTransformers | Embeddings + reranking | Free, fast, runs locally |
| FAISS | Semantic vector search | Free, no server setup needed |
| rank_bm25 | Keyword-based search | Lightweight, catches exact terms FAISS may miss |
| pypdf | Reads text from uploaded PDFs | Free, simple PDF text extraction |
| Groq API (LLM) | Generates the actual answer | Free tier, very fast responses |
| FastAPI | Turns our code into a secured web API | Simple, fast, auto-generates docs |
| python-multipart | Lets FastAPI accept file uploads | Required for `/upload` |
| scikit-learn | Cosine similarity for verification | Simple, effective hallucination check |
| PostgreSQL + psycopg2 | Stores every question/answer permanently | Free, reliable, industry-standard |
| slowapi | Rate limiting | Prevents abuse of the API |
| matplotlib | Generates evaluation graphs | Free, standard Python plotting |
| requests | Automates testing our own API | Simple scripting tool |
| pip-audit | Scans for vulnerable dependencies | Free security auditing tool |

---

## How Each Part Works (In Simple Terms)

### 1. Document Ingestion
- PDF text is extracted **page by page** using `pypdf`, so we always know which page any piece of text came from.
- Text is split into chunks (~300–500 characters) — small enough to embed meaningfully, large enough to keep full sentences intact.
- Each chunk is tagged with metadata: chunk ID, PDF name, page number.
- Each chunk is converted into a vector using `all-MiniLM-L6-v2` and stored in **FAISS**. A parallel **BM25 keyword index** is also built from the same chunks.

### 2. Hybrid Retrieval (FAISS + BM25)
- The question is embedded and FAISS returns the top 20 semantically closest chunks (understands *meaning*).
- The same question is also run through BM25, which returns the top 20 chunks by exact keyword match (catches *exact terms*).
- Both result sets are merged and deduplicated — combining "meaning-based" and "exact-term" matching catches cases either method alone would miss (e.g. specific numbers, drug names, medical codes).

### 3. Cross-Encoder Reranking
- All merged candidates are reranked by a cross-encoder model, which reads the question and each chunk together and scores relevance far more precisely than plain vector distance.
- The top 5 reranked chunks are kept as the final context.

### 4. Answer Generation
- The 5 best chunks + the question are combined into a prompt instructing the LLM to answer using the same terms/phrasing as the source, and to follow any length/format instructions in the question.
- Sent to the LLM (via Groq API) to generate the answer.

### 5. Sentence-Level Verification
- The generated answer is split into individual sentences.
- Each sentence is compared against all source sentences (from the retrieved chunks) and matched to its best-supporting one.
- Each sentence gets a status: **Supported**, **Partially Supported**, or **Unsupported**.

### 6. Confidence Engine
- Combines overall retrieval quality with the sentence verification results into one final confidence score.
- **Strict rule:** if even ONE sentence in the answer is "Unsupported," the whole answer is marked **Low confidence** — even if other sentences scored well. This is intentional — the goal is catching hallucination, not inflating the score.
- Final labels: High, Medium, Low.

### 7. "I Don't Know" Fallback
- If the final confidence score is very low (below 35%), the system doesn't present a shaky guess as an answer.
- Instead, it replaces the answer with an honest message: *"I don't have enough reliable information in the uploaded documents to answer this confidently."*
- Evidence and sentence verification are also cleared in this case, so nothing misleading is shown.

### 8. Evidence Extraction
- For every real answer, the single best-supporting sentence is identified and traced back to its exact **source PDF** and **page number**.

### 9. Feedback System
- Every answer is saved with a unique ID.
- Users can submit 👍 (up) or 👎 (down) feedback tied to that specific answer, stored in its own `feedback` table — useful for spotting which kinds of questions the system struggles with over time.

### 10. Strict vs Direct Mode
- **Strict mode (default):** the full RAG pipeline — retrieval, reranking, verification, confidence — answers only from uploaded documents.
- **Direct mode:** skips retrieval entirely and sends the question straight to the LLM with no restriction — useful for demonstrating the difference RAG makes (grounded vs. potentially hallucinated answers).

### 11. Multi-Model Support
- The LLM used for generation isn't fixed — any Groq-hosted model can be specified per request, making it easy to compare model quality/speed without changing code.

### 12. Storage
- Every question, answer, confidence, score, and **response time** is saved to a PostgreSQL table (`chat_history`) with a timestamp — nothing is lost when the server restarts.

---

## Security Features

| Feature | What it does |
|---------|---------------|
| API Key Authentication | Every request to `/ask`, `/upload`, and `/feedback` requires a valid `X-API-Key` header — blocks unauthorized use |
| File Upload Validation | Only accepts real PDF files, and rejects files over 10MB |
| CORS Protection | Restricts which frontend domains are allowed to call the API |
| Rate Limiting | Limits each user to 10 requests/minute on `/ask`, preventing abuse |
| Dependency Vulnerability Scanning | `pip-audit` checks all installed packages for known security issues — currently clean, no known vulnerabilities |
| Secrets Management | API keys and DB passwords are kept in a `.env` file, never hardcoded in code, and never committed to GitHub |

We also manually verified our GitHub commit history contains no leaked secrets — `.env` was never tracked, and only safe placeholder values exist in the shared `.env.example`.

---

## Evaluation — How We Measure Performance

We built two separate evaluation tools, since they measure different things:

**1. Test-Set Evaluation (`evaluate.py` + `analyze_results.py`)**
A fixed list of questions (covering every uploaded PDF — diabetes, hypertension, PHC, asthma — plus an unrelated "trick" question to test honesty) is sent automatically to our own API. Results (confidence, score, response time, cited source) are logged and turned into summary stats and graphs. This shows *designed* reliability — how well the system performs on questions we specifically chose to test it with.

**2. Real-Usage Evaluation (`evaluate_from_history.py`)**
Pulls every question that has actually been asked (stored in PostgreSQL) and generates the same kind of stats and graph from real usage — not just curated test questions. This shows *actual* reliability, based on genuine interactions with the system, including response time.

Both produce a confidence distribution graph and a JSON file with the raw results — useful evidence for a project report.

---

## Current Project Status

| Component | Status |
|-----------|--------|
| Document ingestion (chunking + embeddings) | Done |
| Metadata storage (PDF name, page number, chunk ID) | Done |
| FAISS semantic search | Done |
| BM25 keyword search (Hybrid Search) | Done |
| Cross-encoder reranking | Done |
| Answer generation (LLM via Groq) | Done |
| Sentence-level verification | Done |
| Confidence Engine (combined scoring) | Done |
| "I don't know" fallback for low-confidence answers | Done |
| Evidence extraction (source PDF + page) | Done |
| Feedback system (up/down, linked to each answer) | Done |
| Strict RAG vs Direct LLM toggle | Done |
| Multi-model selection | Done |
| Response time tracking | Done |
| FastAPI backend | Done |
| PDF upload support (multiple documents) | Done |
| PostgreSQL (chat history + feedback) | Done |
| Security (API key, CORS, rate limiting, file validation) | Done |
| Dependency vulnerability scan | Done — no known issues |
| Evaluation module (test-set based) | Done |
| Evaluation module (real-usage based) | Done |
| React frontend | Not started |
| OCR for images/diagrams in PDFs | Not started (evaluated, decided to skip for now) |

---

## Files In This Project

| File | What it's for |
|------|----------------|
| `api.py` | The actual running application — all features live here |
| `ingest.py` | One-time script to build the initial vector store from a PDF |
| `evaluate.py` | Runs a fixed test-question set against the API and saves results |
| `analyze_results.py` | Turns `evaluate.py`'s results into summary stats and graphs |
| `evaluate_from_history.py` | Pulls real usage data from PostgreSQL and generates stats/graphs |
| `test_questions.py` | The list of test questions used by `evaluate.py` |
| `main.py`, `retriever.py`, `generator.py` | Early standalone versions used while building the pipeline stage by stage — no longer used, kept only as reference (each has a comment at the top explaining this) |

---

## Important Notes

- **We didn't train any AI model.** We use pre-trained models (SentenceTransformers, cross-encoder, the LLM) and built a smart *system* around them — that's the whole idea of RAG.
- **Never run `ingest.py` after uploading documents via the API** — it rebuilds the vector store from scratch and wipes uploaded content. Use `/upload` for adding documents after the initial setup.
- Confidence scores of 55–75% on well-answered, correctly grounded questions are normal — a RAG system that scores 95%+ on every answer is usually just copy-pasting text instead of genuinely generating a helpful response. Hybrid search + reranking has pushed our typical scores into the 80–90% range while keeping answers natural, not copy-pasted.
- OCR (extracting text from diagrams/images inside PDFs) was considered but intentionally not implemented — it requires external software installs with real setup risk, and our current PDFs didn't need it.
- Everything used is **completely free** — no paid tools or subscriptions.

---

## Full List of Installed Tools (What Each One Is For)

```
langchain, langchain-text-splitters   → chunking documents
sentence-transformers                 → embeddings + cross-encoder reranking
faiss-cpu                             → semantic vector search
rank_bm25                             → keyword-based search (hybrid retrieval)
groq                                  → connects to the LLM
python-dotenv                         → loads secrets from .env safely
scikit-learn                          → cosine similarity for verification
fastapi, uvicorn                      → the web API and its server
python-multipart                      → enables file uploads in FastAPI
pypdf                                 → extracts text from PDFs
psycopg2-binary                       → connects to PostgreSQL
slowapi                               → rate limiting
matplotlib                            → evaluation graphs
requests                              → automates testing our own API
pip-audit                             → scans for vulnerable dependencies
```

---

## Installation History (In the Order We Actually Ran Them)

```bash
python -m venv venv
venv\Scripts\activate
```
Creates and activates an isolated environment for this project's tools.

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
Groq's SDK — lets our code call the LLM over their API.

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

```bash
pip install rank_bm25
```
Adds keyword-based search (BM25), used alongside FAISS for hybrid retrieval.

```bash
pip install slowapi
```
Adds rate limiting to protect the API from being spammed.

```bash
pip install pip-audit
```
Not a project dependency — a one-time security tool used to scan all installed packages for known vulnerabilities.

Note: `sentence-transformers` (already installed) also provides the `CrossEncoder` class used for reranking — no separate install was needed for that.

---

## Commands Used Day-to-Day (Running the Project)

```bash
venv\Scripts\activate
```
Activates the virtual environment — run this first, every time, before anything else.

```bash
python ingest.py
```
Builds the vector store from a PDF. Only needs to be run once, at the very start (or if the vector store is deleted and needs rebuilding).

```bash
uvicorn api:app --reload
```
Starts the actual application (the API server). Must be left running in its own terminal — `--reload` restarts it automatically whenever the code changes.

```bash
python evaluate.py
```
Sends a fixed set of test questions to the running API and saves the results. Requires `uvicorn` to be running in a separate terminal first.

```bash
python analyze_results.py
```
Turns `evaluate.py`'s output into summary statistics and graphs.

```bash
python evaluate_from_history.py
```
Pulls every real question ever asked (from PostgreSQL) and generates stats/graphs from actual usage.

```bash
pip-audit
```
Re-run any time to check whether newly installed or updated packages have known vulnerabilities.

---

## Quick Setup Guide

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install langchain langchain-text-splitters sentence-transformers faiss-cpu rank_bm25 groq python-dotenv scikit-learn fastapi uvicorn pypdf psycopg2-binary python-multipart slowapi matplotlib requests
   ```
3. Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   DATABASE_PASSWORD=your_postgres_password_here
   APP_API_KEY=your_custom_api_key_here
   TRANSFORMERS_VERBOSITY=error
   ```
4. Install PostgreSQL, then create the database and tables:
   ```sql
   CREATE DATABASE clinical_rag;

   CREATE TABLE chat_history (
       id SERIAL PRIMARY KEY,
       question TEXT,
       answer TEXT,
       confidence TEXT,
       score FLOAT,
       response_time FLOAT,
       created_at TIMESTAMP DEFAULT NOW()
   );

   CREATE TABLE feedback (
       id SERIAL PRIMARY KEY,
       chat_id INT REFERENCES chat_history(id),
       feedback TEXT CHECK (feedback IN ('up', 'down')),
       created_at TIMESTAMP DEFAULT NOW()
   );
   ```
5. Build the base vector store once:
   ```bash
   python ingest.py
   ```
   Note: since `chunks.pkl` and `vector_store.index` are no longer tracked in GitHub, teammates cloning the repo must run this step themselves before starting the API.
6. Run the API:
   ```bash
   uvicorn api:app --reload
   ```
7. Open `http://127.0.0.1:8000/docs`, click **Authorize**, enter your `APP_API_KEY`, then:
   - Use `POST /upload` to add PDF documents
   - Use `POST /ask` to ask questions (optionally pass `"mode": "direct"` or a different `"model"`)
   - Use `POST /feedback` to submit up/down feedback on a given `chat_id`
8. To evaluate performance:
   ```bash
   python evaluate.py                  # test-set based
   python analyze_results.py
   python evaluate_from_history.py     # real-usage based
   ```
