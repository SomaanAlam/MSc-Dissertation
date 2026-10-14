"""
Module 1: Preprocess BioASQ Dataset for Extractive QA
Cleans and formats BioASQ data into train/val/test splits
Handles factoid and list question types
"""

import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
input_path = Path("data/raw/BioASQ/training13b.json")
output_dir = Path("data/processed/")
output_dir.mkdir(parents=True, exist_ok=True)

# Load raw BioASQ data
with open(input_path) as f:
    raw_data = json.load(f)

examples = []
skipped_no_answer_in_context = 0
skipped_too_long = 0

for item in raw_data["questions"]:
    if item["type"] not in ["factoid", "list"]:
        continue  # Skip yes-no and summary questions

    question = item["body"].strip()
    snippets = [s["text"] for s in item.get("snippets", []) if s.get("text")]
    answers = item.get("exact_answer", [])

    if not snippets or not answers:
        continue  # Require both snippets and at least one answer

    # Handle list vs factoid answer types
    if isinstance(answers, list):
        if isinstance(answers[0], list):
            # list question: answers = [["a", "b"]]
            answer_candidates = answers[0]
        else:
            # factoid: answers = ["EGFR"]
            answer_candidates = answers
    else:
        answer_candidates = [answers]

    # Find snippet containing answer
    best_snippet = None
    best_answer = None
    
    for snippet in snippets:
        for answer in answer_candidates:
            # Check if answer appears in snippet (case-insensitive)
            if answer.lower() in snippet.lower():
                best_snippet = snippet
                best_answer = answer
                break
        if best_snippet:
            break
    
    # If no snippet contains answer, try combining first 2-3 snippets
    if not best_snippet:
        # Combine snippets until we find answer or exceed 400 tokens
        combined = ""
        for snippet in snippets[:3]:
            test_combined = combined + " " + snippet if combined else snippet
            # Rough token estimate (4 chars per token)
            if len(test_combined) < 1600:  # ~400 tokens
                combined = test_combined
                # Check if any answer is now present
                for answer in answer_candidates:
                    if answer.lower() in combined.lower():
                        best_snippet = combined
                        best_answer = answer
                        break
                if best_snippet:
                    break
    
    if not best_snippet or not best_answer:
        skipped_no_answer_in_context += 1
        continue
    
    # Check final length
    if len(best_snippet) > 2000:  # ~500 tokens
        skipped_too_long += 1
        continue

    examples.append({
        "id": item.get("id", ""),
        "question": question,
        "context": best_snippet.strip(),
        "answer": best_answer.strip()
    })

print(f" Collected {len(examples)} extractive QA examples from BioASQ")
print(f"  Skipped {skipped_no_answer_in_context} (answer not in context)")
print(f"  Skipped {skipped_too_long} (context too long)")

# Split into train/val/test with stratification by question type
train, temp = train_test_split(examples, test_size=0.2, random_state=42, shuffle=True)
val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

print(f"\nSplit sizes:")
print(f"  Train: {len(train)}")
print(f"  Val:   {len(val)}")
print(f"  Test:  {len(test)}")

# Save
with open(output_dir / "bioasq_train.json", "w") as f:
    json.dump(train, f, indent=2)

with open(output_dir / "bioasq_val.json", "w") as f:
    json.dump(val, f, indent=2)

with open(output_dir / "bioasq_test.json", "w") as f:
    json.dump(test, f, indent=2)

print("\n Saved to data/processed/")
