
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
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)

print("Answer:", response.choices[0].message.content)