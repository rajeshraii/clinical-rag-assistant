from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load text file
from pypdf import PdfReader

reader = PdfReader("Diabetes_file.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = []  # now stores dictionaries instead of plain strings
chunk_id = 0
for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    page_chunks = splitter.split_text(page_text)
    for c in page_chunks:
        chunks.append({
            "chunk_id": chunk_id,
            "pdf_name": "Diabetes_file.pdf",
            "page_number": page_num,
            "text": c
        })
        chunk_id += 1

# Convert to vectors
model = SentenceTransformer("all-MiniLM-L6-v2")
texts_only = [c["text"] for c in chunks]
embeddings = model.encode(texts_only)

# Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and chunks
faiss.write_index(index, "vector_store.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector store created successfully!")