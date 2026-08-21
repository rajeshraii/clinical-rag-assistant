from fastapi import FastAPI
from pydantic import BaseModel
import os, numpy as np, faiss, pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from fastapi import UploadFile, File
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import shutil
import re
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request


load_dotenv()

API_KEY = os.getenv("APP_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
load_dotenv()

conn = psycopg2.connect(
    host="127.0.0.1",
    database="clinical_rag",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # future React frontend's URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

index = faiss.read_index("vector_store.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/ask", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def ask(request: Request, query: Query):
    query_vector = embedder.encode([query.question])
    D, I = index.search(np.array(query_vector), k=20)  # retrieve more candidates first
    candidates = [chunks[i] for i in I[0]]

    # Rerank using cross-encoder
    pairs = [[query.question, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # Sort candidates by rerank score, keep top 5
    ranked = [c for _, c in sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)]
    retrieved = ranked[:5]

    print("Reranked chunks:", [r["text"] for r in retrieved])
    context = "\n".join([r["text"] for r in retrieved])
    prompt = prompt = prompt = f"Answer the question using the same key terms and phrasing as the context wherever possible. Follow any length or format instructions in the question exactly.\nContext: {context}\nQuestion: {query.question}\nAnswer:"
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    answer_vector = embedder.encode([answer])

    # Break ALL retrieved chunks into individual sentences
    all_sentences = []
    for r in retrieved:
        sentences = re.split(r'(?<=[.!?]) +', r["text"])
        all_sentences.extend(sentences)

    # Remove very short/empty fragments
    all_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 15]

    # Compare answer against each sentence, take the best match
    sentence_vectors = embedder.encode(all_sentences)
    sentence_similarities = cosine_similarity(answer_vector, sentence_vectors)[0]
    similarity = float(max(sentence_similarities))

    # Find which chunk had the best match (for evidence)
    best_match_index = int(np.argmax(sentence_similarities))
    best_sentence = all_sentences[best_match_index]

    # Find which retrieved chunk this sentence came from
    source_chunk = None
    for r in retrieved:
        if best_sentence in r["text"]:
            source_chunk = r
            break

    evidence = {
        "source_pdf": source_chunk["pdf_name"] if source_chunk else "Unknown",
        "page_number": source_chunk["page_number"] if source_chunk else "Unknown",
        "supporting_text": best_sentence
    }


    # Break the ANSWER into sentences
    answer_sentences = re.split(r'(?<=[.!?]) +', answer)
    answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 10]

    sentence_verification = []
    for a_sent in answer_sentences:
        a_vector = embedder.encode([a_sent])
        sims = cosine_similarity(a_vector, sentence_vectors)[0]
        best_score = float(max(sims))
        best_idx = int(np.argmax(sims))

        status = "Supported" if best_score >= 0.55 else "Partially Supported" if best_score >= 0.35 else "Unsupported"

        sentence_verification.append({
            "sentence": a_sent,
            "support_score": round(best_score * 100, 2),
            "status": status,
            "matched_evidence": all_sentences[best_idx]
        })


    # Confidence Engine — combines retrieval quality + per-sentence verification
    retrieval_quality = float(np.mean([s for s in sentence_similarities]))  # how relevant were retrieved chunks overall

    supported_count = sum(1 for s in sentence_verification if s["status"] == "Supported")
    partial_count = sum(1 for s in sentence_verification if s["status"] == "Partially Supported")
    unsupported_count = sum(1 for s in sentence_verification if s["status"] == "Unsupported")
    total_sentences = len(sentence_verification)

    verification_score = (supported_count + 0.5 * partial_count) / total_sentences if total_sentences > 0 else 0

    # Weighted combination: verification matters most, retrieval quality is a secondary signal
    final_confidence_score = (0.7 * verification_score) + (0.3 * retrieval_quality)

    if unsupported_count > 0:
        confidence = "Low"
    elif final_confidence_score >= 0.80:
        confidence = "High"
    else:
        confidence = "Low"

    cursor.execute(
        "INSERT INTO chat_history (question, answer, confidence, score) VALUES (%s, %s, %s, %s)",
        (query.question, answer, confidence, round(final_confidence_score * 100, 2))
    )
    conn.commit()

    return {
    "answer": answer,
    "confidence": confidence,
    "score": round(final_confidence_score * 100, 2),
    "evidence": evidence,
    "sentence_verification": sentence_verification
}

@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Validate file size (limit to 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    await file.seek(0)  # reset file pointer after reading

    processed_files = "processed.txt"
    # ... rest of your existing code continues here

    processed_files = "processed.txt"
    if os.path.exists(processed_files):
        with open(processed_files) as f:
            done = f.read().splitlines()
        if file.filename in done:
            return {"message": "File already processed, skipping"}

    # Save uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    global chunks
    next_id = len(chunks)  # continue numbering from existing chunks
    new_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        page_chunks = splitter.split_text(page_text)
        for c in page_chunks:
            new_chunks.append({
                "chunk_id": next_id,
                "pdf_name": file.filename,
                "page_number": page_num,
                "text": c
            })
            next_id += 1

    texts_only = [c["text"] for c in new_chunks]
    new_embeddings = embedder.encode(texts_only)
    index.add(np.array(new_embeddings))

    chunks.extend(new_chunks)
    faiss.write_index(index, "vector_store.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return {"message": f"{file.filename} uploaded and processed successfully", "chunks_added": len(new_chunks)}

    with open(processed_files, "a") as f:
        f.write(file.filename + "\n")

    return {"message": f"{file.filename} uploaded and processed successfully", "chunks_added": len(new_chunks)}