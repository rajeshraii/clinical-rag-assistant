"""
NOTE: This file is no longer used in the running application.

Why it was created: Standalone script to test Stage 3 (answer generation) in
isolation — sending a hardcoded context and question to the LLM and printing
the answer, before generation was combined with retrieval.

Why it's no longer used: Answer generation now happens inside api.py's /ask
endpoint, using dynamically retrieved and reranked context instead of a
hardcoded one. Kept only as a reference of the early, simpler generation logic.
"""

import os
from dotenv import load_dotenv
from groq import Groq

# Initialize Groq client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Retrieved chunks (from retriever)
context = """
Symptoms include frequent urination, excessive thirst, and blurred vision.
Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin.
Type 1 diabetes is caused by an autoimmune reaction.
"""

# User question
query = "What are the symptoms of diabetes?"

# Build prompt
prompt = f"""Answer the question based only on the context below.
Context: {context}
Question: {query}
Answer:"""

# Generate answer
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": prompt}]
)

print("Answer:", response.choices[0].message.content)