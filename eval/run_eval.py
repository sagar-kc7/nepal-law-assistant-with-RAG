import json
import requests

API_URL = "http://localhost:8000/search"

def load_eval_set(path="eval/eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_retrieved_articles(question: str):
    response = requests.post(API_URL, json={"question": question})
    response.raise_for_status()
    data = response.json()
    return [chunk["article"] for chunk in data["chunks"]]

def compute_precision_recall(expected_ids, retrieved_ids):
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)
    overlap = expected_set & retrieved_set

    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(expected_set) if expected_set else 0.0
    return precision, recall

def evaluate(eval_set):
    results = []
    for item in eval_set:
        retrieved_articles = get_retrieved_articles(item["question"])
        precision, recall = compute_precision_recall(
            item["expected_articles"], retrieved_articles
        )
        results.append({
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "expected_articles": item["expected_articles"],
            "retrieved_articles": retrieved_articles,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
        })
    return results

def print_report(results):
    print(f"{'Type':<15} {'Precision':<10} {'Recall':<10} Question")
    print("-" * 70)
    for r in results:
        print(f"{r['type']:<15} {r['precision']:<10} {r['recall']:<10} {r['question'][:40]}")

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    print("-" * 70)
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall:    {avg_recall:.2f}")

if __name__ == "__main__":
    eval_set = load_eval_set()
    results = evaluate(eval_set)
    print_report(results)