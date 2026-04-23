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
D, I = index.search(np.array(query_vector), k=3)
context = "\n".join([chunks[i] for i in I[0]])

# Generate answer
prompt = f"""Answer the question based only on the context below.
Context: {context}
Question: {query}
Answer:"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:", response.choices[0].message.content)