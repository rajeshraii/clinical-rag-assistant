from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load saved data
index = faiss.read_index("vector_store.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask a question
query = "What are the symptoms of diabetes?"

# Convert question to vector
query_vector = model.encode([query])

# Search FAISS
D, I = index.search(np.array(query_vector), k=3)

# Show results
print("Top relevant chunks:")
for i in I[0]:
    print(f"- {chunks[i]}")