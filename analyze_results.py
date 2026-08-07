import json
import matplotlib.pyplot as plt

with open("evaluation_results.json") as f:
    results = json.load(f)

# Summary stats
total = len(results)
high = sum(1 for r in results if r["confidence"] == "High")
medium = sum(1 for r in results if r["confidence"] == "Medium")
low = sum(1 for r in results if r["confidence"] == "Low")
avg_score = sum(r["score"] for r in results) / total
avg_time = sum(r["response_time"] for r in results) / total

print(f"Total Questions: {total}")
print(f"High Confidence: {high} ({round(high/total*100,1)}%)")
print(f"Medium Confidence: {medium} ({round(medium/total*100,1)}%)")
print(f"Low Confidence: {low} ({round(low/total*100,1)}%)")
print(f"Average Score: {round(avg_score,2)}%")
print(f"Average Response Time: {round(avg_time,2)}s")

# Bar chart - Confidence distribution
plt.figure(figsize=(6,4))
plt.bar(["High", "Medium", "Low"], [high, medium, low], color=["green", "orange", "red"])
plt.title("Confidence Level Distribution")
plt.ylabel("Number of Questions")
plt.savefig("confidence_distribution.png")
plt.show()

# Bar chart - Score per question
plt.figure(figsize=(10,5))
questions = [r["question"][:20] + "..." for r in results]
scores = [r["score"] for r in results]
plt.barh(questions, scores, color="skyblue")
plt.xlabel("Score (%)")
plt.title("Score per Question")
plt.tight_layout()
plt.savefig("score_per_question.png")
plt.show()