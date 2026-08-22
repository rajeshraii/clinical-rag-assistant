"""
NOTE: This file is no longer used in the running application.

Why it was created: Early combined pipeline (retrieval + generation + validation)
built to test the full RAG flow end-to-end from the terminal, before the project
had a web API.

Why it's no longer used: All of this logic was later moved into api.py, which
wraps the same pipeline in a FastAPI backend — adding hybrid search, reranking,
sentence verification, confidence scoring, evidence extraction, and PostgreSQL
storage on top of it. api.py is the actual application now; this file is kept
only as a reference of how the pipeline looked at that stage.
"""

import os
import numpy as np
import faiss
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Load saved data
index = faiss.read_index("vector_store.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# User question
query = input("Ask a question: ")

# Retrieve relevant chunks
query_vector = embedder.encode([query])
D, I = index.search(np.array(query_vector), k=5)
context = "\n".join([chunks[i] for i in I[0]])

# Generate answer
prompt = f"""Answer the question based only on the context below.
Context: {context}
Question: {query}
Answer:"""

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:", response.choices[0].message.content)

# Validation
answer_vector = embedder.encode([response.choices[0].message.content])
context_vector = embedder.encode([context])
similarity = cosine_similarity(answer_vector, context_vector)[0][0]

score = round(float(similarity) * 100, 2)
print(f"Validation Score: {score}%")

if similarity > 0.5:
    print("Status:  Answer is reliable")
else:
    print("Status:  Answer may not be reliable")


# Confidence Score
if similarity >= 0.75:
    confidence = "High"
    emoji = "🟢"
elif similarity >= 0.50:
    confidence = "Medium"
    emoji = "🟡"
else:
    confidence = "Low"
    emoji = "🔴"

print(f"Confidence: {emoji} {confidence} ({score}%)")

print("Retrieved chunks:")
for i in I[0]:
    print(f"- {chunks[i]}")