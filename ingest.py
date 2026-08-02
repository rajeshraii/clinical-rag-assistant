from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load text file
from pypdf import PdfReader

reader = PdfReader("Diabetes_file.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")
print("Chunks:", chunks)

# Convert to vectors
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and chunks
faiss.write_index(index, "vector_store.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector store created successfully!")