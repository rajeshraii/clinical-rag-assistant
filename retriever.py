"""
NOTE: This file is no longer used in the running application.

Why it was created: Standalone script to test Stage 2 (retrieval) in isolation —
taking a hardcoded question, searching FAISS, and printing the matching chunks —
before retrieval was combined with the rest of the pipeline.

Why it's no longer used: Retrieval now happens inside api.py's /ask endpoint,
combined with BM25 hybrid search and cross-encoder reranking, which this file
does not include. Kept only as a reference of the early, simpler retrieval logic.
"""

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