from fastapi import FastAPI
from pydantic import BaseModel
import os, numpy as np, faiss, pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from fastapi import UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import shutil
import re
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

index = faiss.read_index("vector_store.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    query_vector = embedder.encode([query.question])
    D, I = index.search(np.array(query_vector), k=7)
    print("Retrieved chunks:", [chunks[i] for i in I[0]])
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = prompt = prompt = f"Answer the question using the same key terms and phrasing as the context wherever possible. Follow any length or format instructions in the question exactly.\nContext: {context}\nQuestion: {query.question}\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    answer_vector = embedder.encode([answer])

    # Break ALL retrieved chunks into individual sentences
    all_sentences = []
    for i in I[0]:
        sentences = re.split(r'(?<=[.!?]) +', chunks[i])
        all_sentences.extend(sentences)

    # Remove very short/empty fragments
    all_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 15]

    # Compare answer against each sentence, take the best match
    sentence_vectors = embedder.encode(all_sentences)
    sentence_similarities = cosine_similarity(answer_vector, sentence_vectors)[0]
    similarity = float(max(sentence_similarities))

    confidence = "High" if similarity >= 0.75 else "Medium" if similarity >= 0.50 else "Low"

    cursor.execute(
    "INSERT INTO chat_history (question, answer, confidence, score) VALUES (%s, %s, %s, %s)",
    (query.question, answer, confidence, round(similarity * 100, 2))
    )
    conn.commit()

    return {"answer": answer, "confidence": confidence, "score": round(similarity * 100, 2)}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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

    # Extract text from PDF
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Chunk it
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    new_chunks = splitter.split_text(text)

    # Embed and add to FAISS
    new_embeddings = embedder.encode(new_chunks)
    index.add(np.array(new_embeddings))

    # Update chunks list and save
    global chunks
    chunks.extend(new_chunks)
    faiss.write_index(index, "vector_store.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return {"message": f"{file.filename} uploaded and processed successfully", "chunks_added": len(new_chunks)}

    with open(processed_files, "a") as f:
        f.write(file.filename + "\n")

    return {"message": f"{file.filename} uploaded and processed successfully", "chunks_added": len(new_chunks)}