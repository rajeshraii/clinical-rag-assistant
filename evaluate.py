import requests
import time
import json
import os
from dotenv import load_dotenv
import time


from test_questions import test_questions

results = []

for q in test_questions:
    start_time = time.time()
    load_dotenv()
    headers = {"X-API-Key": os.getenv("APP_API_KEY")}
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": q}, headers=headers)
    elapsed = time.time() - start_time

    data = response.json()
    results.append({
        "question": q,
        "confidence": data.get("confidence"),
        "score": data.get("score"),
        "response_time": round(elapsed, 2),
        "source_pdf": data.get("evidence", {}).get("source_pdf"),
    })
    time.sleep(7)

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete! Results saved to evaluation_results.json")