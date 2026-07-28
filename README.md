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
- Upload a medical PDF
- Ask questions in plain English
- Get an answer that is grounded in that PDF (not made up)
- See a confidence score (High/Medium/Low) for every answer
- See exactly which part of the document the answer came from

This turns AI from "maybe right, maybe not" into a transparent, trustworthy clinical assistant.

---

## The Big Picture — How It Works (Simple Version)

```
Document → Split into small pieces → Convert to "meaning numbers" → Store
Question → Convert to "meaning numbers" → Find closest matching pieces
Matching pieces + Question → Sent to AI model → Generates answer
Answer → Checked against source → Confidence score assigned
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
| Llama 3 (via Groq API) | Generates the actual answer | Free tier, very fast responses |
| FastAPI | Turns our code into a web API | Simple, fast, auto-generates docs |
| scikit-learn | Checks if answer matches source | Simple way to catch AI hallucination |

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
- **Chunking**: Split the document text into small pieces (~100 characters each). Small pieces work better than giant blocks of text.
- **Embedding**: Convert each chunk into a vector — a list of numbers that represents its *meaning* (not just the words).
- **Storing**: Save all these vectors into FAISS, a database built specifically for fast vector searching.

Think of it like: turning a book into a searchable index, but the index understands meaning, not just keywords.

### Step 3 — Retriever (`retriever.py`)
This is how we find the right information for a question:
- Take the user's question, convert it into a vector the same way.
- Ask FAISS: "which stored chunks are closest in meaning to this question?"
- FAISS returns the top 3 closest matches.

This works because of how embeddings work: sentences with similar meaning end up mathematically "close" to each other, even if they don't share the same words.

### Step 4 — Answer Generation (`generator.py`)
This is where the AI actually writes an answer:
- We take the matching chunks + the user's question and build an instruction ("prompt") that says: *"Answer only using this information."*
- This prompt is sent to Llama 3 (a large language model) hosted on Groq's free servers.
- Llama 3 sends back a generated answer based only on what we gave it.

### Step 5 — Combining Everything (`main.py`)
This file merges ingestion + retrieval + generation into one live flow that takes a real question from the user and runs it through the whole process automatically.

### Step 6 — Validation Layer (added to `main.py`)
This is our safety check against hallucination:
- We convert the generated answer into a vector, and the source context into a vector.
- We measure the **similarity** between them (called cosine similarity — basically, how closely aligned two things are in meaning).
- High similarity = the answer is well-grounded in the source. Low similarity = the answer might be made up.

### Step 7 — Confidence Score (added to `main.py`)
We turn that similarity number into an easy label:
- 75% and above → 🟢 High confidence
- 50–75% → 🟡 Medium confidence
- Below 50% → 🔴 Low confidence

This gives the user an honest, at-a-glance sense of how trustworthy each answer is.

### Step 8 — FastAPI Backend (`api.py`)
We wrapped the entire pipeline into a web API so it can be accessed over the internet (not just from a terminal). This is what our future React frontend will talk to.

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
```

Test it at: `http://127.0.0.1:8000/docs`

---

## Current Project Status

| Component | Status |
|-----------|--------|
| Document ingestion (chunking + embeddings) | ✅ Done |
| Vector storage & search (FAISS) | ✅ Done |
| Answer generation (Llama 3 via Groq) | ✅ Done |
| Validation layer (cosine similarity) | ✅ Done |
| Confidence scoring | ✅ Done |
| FastAPI backend | ✅ Done |
| PDF upload support | ⏳ Not started |
| React frontend | ⏳ Not started |
| PostgreSQL (chat history) | ⏳ Not started |
| Source citation display | ⏳ Not started |
| Feedback system (👍/👎) | ⏳ Not started |

---

## Important Notes

- **We didn't train any AI model.** We use pre-trained models (SentenceTransformers, Llama 3) and built a smart *system* around them — that's the whole idea of RAG (Retrieval-Augmented Generation).
- **API keys must never be committed to GitHub.** We use a `.env` file (kept out of GitHub via `.gitignore`) to store the Groq API key securely.
- Everything used so far is **completely free** — no paid tools or subscriptions.

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
   pip install langchain langchain-text-splitters sentence-transformers faiss-cpu groq python-dotenv scikit-learn fastapi uvicorn
   ```
4. Create your own `.env` file with:
   ```
   GROQ_API_KEY=your_own_key_here
   ```
5. Run the ingestion once to build the vector store:
   ```bash
   python ingest.py
   ```
6. Run the full pipeline:
   ```bash
   python main.py
   ```
7. Or run it as an API:
   ```bash
   uvicorn api:app --reload
   ```
