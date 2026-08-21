import psycopg2
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json

load_dotenv()

conn = psycopg2.connect(
    host="127.0.0.1",
    database="clinical_rag",
    user="postgres",
    password=os.getenv("DB_PASSWORD")
)
cursor = conn.cursor()

cursor.execute("SELECT question, confidence, score FROM chat_history ORDER BY created_at DESC")
rows = cursor.fetchall()

total = len(rows)
if total == 0:
    print("No history found yet — ask some questions first!")
else:
    high = sum(1 for r in rows if r[1] == "High")
    medium = sum(1 for r in rows if r[1] == "Medium")
    low = sum(1 for r in rows if r[1] == "Low")
    avg_score = sum(r[2] for r in rows) / total


    # Save raw results as JSON
    history_data = [
        {"question": r[0], "confidence": r[1], "score": r[2]}
        for r in rows
        ]
    with open("real_usage_results.json", "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"Total Real Questions Asked: {total}")
    print(f"High Confidence: {high} ({round(high/total*100,1)}%)")
    print(f"Medium Confidence: {medium} ({round(medium/total*100,1)}%)")
    print(f"Low Confidence: {low} ({round(low/total*100,1)}%)")
    print(f"Average Score: {round(avg_score,2)}%")

    # Bar chart - Confidence distribution (from real usage)
    plt.figure(figsize=(6,4))
    plt.bar(["High", "Medium", "Low"], [high, medium, low], color=["green", "orange", "red"])
    plt.title("Real Usage — Confidence Level Distribution")
    plt.ylabel("Number of Questions")
    plt.savefig("real_confidence_distribution.png")
    plt.show()

cursor.close()
conn.close()