import requests
import time
import json

from test_questions import test_questions

results = []

for q in test_questions:
    start_time = time.time()
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": q})
    elapsed = time.time() - start_time

    data = response.json()
    results.append({
        "question": q,
        "confidence": data.get("confidence"),
        "score": data.get("score"),
        "response_time": round(elapsed, 2),
        "source_pdf": data.get("evidence", {}).get("source_pdf"),
    })

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete! Results saved to evaluation_results.json")