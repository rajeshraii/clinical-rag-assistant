from fastapi import FastAPI
from pydantic import BaseModel
import os, numpy as np, faiss, pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    database="clinical_rag",
    user="postgres",
    password=os.getenv("DATABASE_PASSWORD")
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
    D, I = index.search(np.array(query_vector), k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"Answer based only on context.\nContext: {context}\nQuestion: {query.question}\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    answer_vector = embedder.encode([answer])
    context_vector = embedder.encode([context])
    similarity = float(cosine_similarity(answer_vector, context_vector)[0][0])

    confidence = "High" if similarity >= 0.75 else "Medium" if similarity >= 0.50 else "Low"

    cursor.execute(
    "INSERT INTO chat_history (question, answer, confidence, score) VALUES (%s, %s, %s, %s)",
    (query.question, answer, confidence, round(similarity * 100, 2))
    )
    conn.commit()

    return {"answer": answer, "confidence": confidence, "score": round(similarity * 100, 2)}