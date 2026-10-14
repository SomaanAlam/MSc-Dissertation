"""
Module 2: Data Quality Analysis for BioASQ Dataset

Comprehensive statistics and example inspections
Comparative analysis across train/val/test splits

"""

import json
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_dataset(file_path):
    """Comprehensive analysis of BioASQ dataset
    Args:
        file_path: Path to dataset JSON file
    Returns:
        data: Loaded dataset for further inspection"""
    
    with open(file_path) as f:
        data = json.load(f)
    
    print("="*60)
    print(f"DATASET ANALYSIS: {Path(file_path).name}")
    print("="*60)
    
    # Basic statistics
    print(f"\n[1] Basic Statistics:")
    print(f"    Total examples: {len(data)}")
    
    # Context length analysis
    context_lengths = [len(ex['context']) for ex in data]
    print(f"\n[2] Context Length (characters):")
    print(f"    Mean:   {sum(context_lengths)/len(context_lengths):.1f}")
    print(f"    Min:    {min(context_lengths)}")
    print(f"    Max:    {max(context_lengths)}")
    print(f"    Median: {sorted(context_lengths)[len(context_lengths)//2]}")
    
    # Rough token estimate (4 chars per token)
    token_estimates = [l/4 for l in context_lengths]
    print(f"\n[3] Estimated Token Count:")
    print(f"    Mean:   {sum(token_estimates)/len(token_estimates):.1f}")
    print(f"    > 400 tokens: {sum(1 for t in token_estimates if t > 400)} ({sum(1 for t in token_estimates if t > 400)/len(token_estimates)*100:.1f}%)")
    print(f"    > 450 tokens: {sum(1 for t in token_estimates if t > 450)} ({sum(1 for t in token_estimates if t > 450)/len(token_estimates)*100:.1f}%)")
    
    # Question length
    question_lengths = [len(ex['question']) for ex in data]
    print(f"\n[4] Question Length (characters):")
    print(f"    Mean:   {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"    Min:    {min(question_lengths)}")
    print(f"    Max:    {max(question_lengths)}")
    
    # Answer length
    answer_lengths = [len(ex['answer']) for ex in data]
    print(f"\n[5] Answer Length (characters):")
    print(f"    Mean:   {sum(answer_lengths)/len(answer_lengths):.1f}")
    print(f"    Min:    {min(answer_lengths)}")
    print(f"    Max:    {max(answer_lengths)}")
    print(f"    Single word: {sum(1 for a in answer_lengths if a < 20)} ({sum(1 for a in answer_lengths if a < 20)/len(answer_lengths)*100:.1f}%)")
    
    # Answer position in context
    positions = []
    for ex in data:
        pos = ex['context'].lower().find(ex['answer'].lower())
        if pos != -1:
            relative_pos = pos / len(ex['context'])
            positions.append(relative_pos)
    
    print(f"\n[6] Answer Position in Context:")
    print(f"    Mean position:   {sum(positions)/len(positions):.2%}")
    print(f"    At start (0-20%): {sum(1 for p in positions if p < 0.2)/len(positions)*100:.1f}%")
    print(f"    In middle (20-80%): {sum(1 for p in positions if 0.2 <= p <= 0.8)/len(positions)*100:.1f}%")
    print(f"    At end (80-100%): {sum(1 for p in positions if p > 0.8)/len(positions)*100:.1f}%")
    
    # Answer verification
    print(f"\n[7] Answer Verification:")
    not_found = 0
    case_mismatch = 0
    for ex in data:
        if ex['answer'] not in ex['context']:
            if ex['answer'].lower() in ex['context'].lower():
                case_mismatch += 1
            else:
                not_found += 1
    
    print(f"    Exact match:     {len(data) - not_found - case_mismatch} ({(len(data) - not_found - case_mismatch)/len(data)*100:.1f}%)")
    print(f"    Case mismatch:   {case_mismatch} ({case_mismatch/len(data)*100:.1f}%)")
    print(f"    Not found:       {not_found} ({not_found/len(data)*100:.1f}%)")
    
    # Relevance scores (if available)
    if 'relevance_score' in data[0]:
        scores = [ex['relevance_score'] for ex in data]
        print(f"\n[8] Relevance Scores:")
        print(f"    Mean:   {sum(scores)/len(scores):.3f}")
        print(f"    Min:    {min(scores):.3f}")
        print(f"    Max:    {max(scores):.3f}")
        print(f"    < 0.3:  {sum(1 for s in scores if s < 0.3)} ({sum(1 for s in scores if s < 0.3)/len(scores)*100:.1f}%)")
        print(f"    > 0.5:  {sum(1 for s in scores if s > 0.5)} ({sum(1 for s in scores if s > 0.5)/len(scores)*100:.1f}%)")
    
    return data

def show_examples(data, n=5):
    """Show random examples for manual inspection"""
    
    print(f"\n{'='*60}")
    print(f"RANDOM EXAMPLES FOR MANUAL INSPECTION")
    print(f"{'='*60}")
    
    samples = random.sample(data, min(n, len(data)))
    
    for i, ex in enumerate(samples, 1):
        print(f"\n--- Example {i} ---")
        print(f"ID:       {ex.get('id', 'N/A')}")
        print(f"Question: {ex['question']}")
        print(f"Answer:   '{ex['answer']}'")
        print(f"Context:  {ex['context'][:200]}...")
        
        # Check if answer is in context
        if ex['answer'].lower() in ex['context'].lower():
            pos = ex['context'].lower().find(ex['answer'].lower())
            start = max(0, pos - 50)
            end = min(len(ex['context']), pos + len(ex['answer']) + 50)
            print(f"Found at: ...{ex['context'][start:end]}...")
        else:
            print(f"WARNING: Answer not found in context!")
        
        if 'relevance_score' in ex:
            print(f"Score:    {ex['relevance_score']:.3f}")

def compare_splits(train_file, val_file, test_file):
    """Compare statistics across splits"""
    
    print(f"\n{'='*60}")
    print("SPLIT COMPARISON")
    print(f"{'='*60}")
    
    with open(train_file) as f:
        train = json.load(f)
    with open(val_file) as f:
        val = json.load(f)
    with open(test_file) as f:
        test = json.load(f)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    
    # Compare answer lengths
    train_ans_len = sum(len(ex['answer']) for ex in train) / len(train)
    val_ans_len = sum(len(ex['answer']) for ex in val) / len(val)
    test_ans_len = sum(len(ex['answer']) for ex in test) / len(test)
    
    print(f"\nAverage answer length:")
    print(f"  Train: {train_ans_len:.1f}")
    print(f"  Val:   {val_ans_len:.1f}")
    print(f"  Test:  {test_ans_len:.1f}")
    
    # Compare context lengths
    train_ctx_len = sum(len(ex['context']) for ex in train) / len(train)
    val_ctx_len = sum(len(ex['context']) for ex in val) / len(val)
    test_ctx_len = sum(len(ex['context']) for ex in test) / len(test)
    
    print(f"\nAverage context length:")
    print(f"  Train: {train_ctx_len:.1f}")
    print(f"  Val:   {val_ctx_len:.1f}")
    print(f"  Test:  {test_ctx_len:.1f}")

def main():
    """Main execution"""
    
    base_dir = Path("data/processed")
    
    # Analyze each split
    for split in ["train", "val", "test"]:
        file_path = base_dir / f"bioasq_{split}.json"
        if file_path.exists():
            data = analyze_dataset(str(file_path))
            show_examples(data, n=3)
            print("\n")
        else:
            print(f"  {file_path} not found")
    
    # Compare splits
    train_file = base_dir / "bioasq_train.json"
    val_file = base_dir / "bioasq_val.json"
    test_file = base_dir / "bioasq_test.json"
    
    if all(f.exists() for f in [train_file, val_file, test_file]):
        compare_splits(str(train_file), str(val_file), str(test_file))
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("\n Key Insights to Look For:")
    print("  1. Are answers actually in contexts?")
    print("  2. Are contexts too long (> 450 tokens)?")
    print("  3. Are answers at good positions (not at edges)?")
    print("  4. Do questions make sense?")
    print("  5. Are relevance scores reasonable (> 0.3)?")

if __name__ == "__main__":
    main()