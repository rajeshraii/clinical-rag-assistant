# Clinical RAG Assistant

A RAG-based (Retrieval-Augmented Generation) Clinical Decision Support System that answers medical questions using real documents — not guesses.

---

## Why We're Building This

Doctors and students deal with huge amounts of medical documents (research papers, guidelines, patient notes). Two big problems exist today:

1. **Normal search is dumb** — keyword search doesn't understand meaning, so it often misses relevant info.
2. **Normal AI chatbots hallucinate** — they can confidently give wrong medical information, which is dangerous in healthcare.

**Our solution:** A system that only answers using the documents you give it, and tells you *how confident* it is in each answer — so nothing is a blind guess.

---

## What The Final Outcome Will Be

A web app where a user can:
- Upload any medical PDF
- Ask questions in plain English
- Get an answer that is grounded in that PDF (not made up)
- See a confidence score (High/Medium/Low) for every answer
- See exactly which part of the document the answer came from
- Have every question, answer, and confidence score saved for later review

This turns AI from "maybe right, maybe not" into a transparent, trustworthy clinical assistant.

---

## The Big Picture — How It Works (Simple Version)

```
PDF Upload → Split into small pieces → Convert to "meaning numbers" → Store
Question → Convert to "meaning numbers" → Find closest matching pieces
Matching pieces + Question → Sent to AI model → Generates answer
Answer → Checked against source → Confidence score assigned → Saved to database
```

That's the whole idea. Everything below is how we actually built it, step by step.

---

## Tech Stack (What We Used and Why)

| Tool | Purpose | Why we chose it |
|------|---------|------------------|
| Python | Main programming language | Best support for AI/ML tools |
| LangChain | Splits documents into chunks | Industry standard, easy to use |
| SentenceTransformers | Converts text into vectors (meaning) | Free, fast, runs on your own PC |
| FAISS | Stores and searches vectors | Free, no server setup needed |
| pypdf | Reads text from uploaded PDFs | Free, simple PDF text extraction |
| Llama 3 (via Groq API) | Generates the actual answer | Free tier, very fast responses |
| FastAPI | Turns our code into a web API | Simple, fast, auto-generates docs |
| python-multipart | Lets FastAPI accept file uploads | Required for the `/upload` endpoint |
| scikit-learn | Checks if answer matches source | Simple way to catch AI hallucination |
| PostgreSQL + psycopg2 | Stores every question/answer permanently | Free, reliable, industry-standard database |

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
- **Chunking**: Split the document text into small pieces. Small pieces work better than giant blocks of text — but too small breaks sentences apart, so we tuned this to `chunk_size=300-500` after testing.
- **Embedding**: Convert each chunk into a vector — a list of numbers that represents its *meaning* (not just the words).
- **Storing**: Save all these vectors into FAISS, a database built specifically for fast vector searching.

Think of it like: turning a book into a searchable index, but the index understands meaning, not just keywords.

**Note:** `ingest.py` is only meant to be run ONCE, at the very beginning, to build the base vector store. After that, new documents are added through the `/upload` API endpoint instead — running `ingest.py` again will wipe and rebuild everything from scratch.

### Step 3 — Retriever (`retriever.py`)
This is how we find the right information for a question:
- Take the user's question, convert it into a vector the same way.
- Ask FAISS: "which stored chunks are closest in meaning to this question?"
- FAISS returns the top closest matches (we use `k=7` — tuned up from `k=3` for better context coverage).

This works because of how embeddings work: sentences with similar meaning end up mathematically "close" to each other, even if they don't share the same words.

### Step 4 — Answer Generation (`generator.py`)
This is where the AI actually writes an answer:
- We take the matching chunks + the user's question and build an instruction ("prompt") that says: *"Answer using the same key terms and phrasing as the context, and follow any length/format instructions in the question."*
- This prompt is sent to Llama 3 (a large language model) hosted on Groq's free servers.
- Llama 3 sends back a generated answer based only on what we gave it.

### Step 5 — Combining Everything (`main.py`)
This file merges ingestion + retrieval + generation into one live flow that takes a real question from the user and runs it through the whole process automatically.

### Step 6 — Validation Layer (in `main.py` / `api.py`)
This is our safety check against hallucination:
- We convert the generated answer into a vector.
- Instead of comparing it against all retrieved chunks mixed together (which dilutes the score), we split the retrieved chunks into individual **sentences** and compare the answer against each sentence separately, taking the **best match**.
- This gives a more honest, precise measure of whether the answer is genuinely grounded in a specific part of the source document.

### Step 7 — Confidence Score
We turn that similarity number into an easy label:
- 75% and above → High confidence
- 50–75% → Medium confidence
- Below 50% → Low confidence

This gives the user an honest, at-a-glance sense of how trustworthy each answer is.

### Step 8 — FastAPI Backend (`api.py`)
We wrapped the entire pipeline into a web API so it can be accessed over the internet (not just from a terminal). This is what our future React frontend will talk to.

```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload
```

Test it at: `http://127.0.0.1:8000/docs`

### Step 9 — PDF Upload Support (`/upload` endpoint in `api.py`)
Instead of manually editing a `.txt` file, users can now upload any medical PDF directly through the API:
- The PDF is saved into an `uploads/` folder
- Text is extracted using `pypdf`
- The text is chunked, embedded, and added to the existing FAISS store (without deleting what's already there)
- A duplicate-check prevents the same file from being processed twice
- We've tested this with multiple real PDFs (diabetes, hypertension, asthma fact sheets) covering different topics in the same knowledge base

### Step 10 — PostgreSQL Integration
Every question, answer, confidence level, and score is now saved permanently to a PostgreSQL database (`clinical_rag`), in a table called `chat_history`. This means:
- Nothing is lost when the server restarts
- We can later build analytics or a history view showing past questions and answers
- Database credentials are kept in `.env`, never hardcoded in the code

---

## Current Project Status

| Component | Status |
|-----------|--------|
| Document ingestion (chunking + embeddings) | Done |
| Vector storage & search (FAISS) | Done |
| Answer generation (Llama 3 via Groq) | Done |
| Validation layer (best-match sentence similarity) | Done |
| Confidence scoring | Done |
| FastAPI backend | Done |
| PDF upload support (multiple documents) | Done |
| PostgreSQL (chat history) | Done |
| React frontend | Not started |
| Source citation display | Not started |
| Feedback system (thumbs up/down) | Not started |

---

## Important Notes

- **We didn't train any AI model.** We use pre-trained models (SentenceTransformers, Llama 3) and built a smart *system* around them — that's the whole idea of RAG (Retrieval-Augmented Generation).
- **API keys and DB passwords must never be committed to GitHub.** We use a `.env` file (kept out of GitHub via `.gitignore`) to store the Groq API key and database password securely.
- **Never run `ingest.py` after uploading documents via the API** — it rebuilds the vector store from scratch and wipes uploaded content. Use `/upload` for adding documents after the initial setup.
- Confidence scores of 55-75% on well-answered, correctly grounded questions are normal and expected — a RAG system that scores 95%+ on every answer is usually just copy-pasting text instead of genuinely generating a helpful response.
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
   pip install langchain langchain-text-splitters sentence-transformers faiss-cpu groq python-dotenv scikit-learn fastapi uvicorn pypdf psycopg2-binary python-multipart
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
