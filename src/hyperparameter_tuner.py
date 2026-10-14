"""
Module 7: Hyperparameter Tuning
Grid search to find optimal retrieval and QA configuration
Tunes on validation set, saves best config for final evaluation

"""

import json
import time
from pathlib import Path
from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

from src.retriever import HybridRetriever
from src.qa_models import QAModel

def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison (SQuAD-style)

    Args:        text: The input text string
    Returns:        Normalized text string
    """
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(prediction: str, ground_truth: str) -> float:
    
    """
    Compute token-level F1 score between prediction and ground truth

    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
    Returns:
        F1 score as float
    """
    
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if len(pred_tokens) == len(truth_tokens) else 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    num_common = len(common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_configuration(retriever, qa_model, val_data, 
                          top_k, rrf_k, qa_top_k):
    """
    Evaluate hyperparameter configuration on validation set
    Args:
        retriever: The HybridRetriever instance
        qa_model: The QAModel instance
        val_data: List of validation examples
        top_k: Number of passages to retrieve
        rrf_k: RRF fusion parameter
        qa_top_k: Number of passages for QA model to process
    Returns:
        dict: Dictionary with 'f1', 'em', 'valid_examples'
    
    """
    
    total_f1 = 0.0
    total_em = 0.0
    valid_examples = 0
    
    # Temporarily update retriever config
    original_rrf_k = retriever.rrf_k
    retriever.rrf_k = rrf_k
    
    for item in val_data:
        # Retrieve passages
        retrieved = retriever.search(item['question'], top_k=top_k)
        
        # Answer with QA model
        answer = qa_model.answer_with_passages(
            item['question'],
            retrieved[:qa_top_k]
        )
        
        # Skip if no answer found
        if answer['no_answer']:
            continue
        
        valid_examples += 1
        
        # Compute metrics
        pred = answer['answer']
        truth = item['answer']
        
        f1 = compute_f1(pred, truth)
        em = 1.0 if normalize_answer(pred) == normalize_answer(truth) else 0.0
        
        total_f1 += f1
        total_em += em
    
    # Restore original config
    retriever.rrf_k = original_rrf_k
    
    # Return average metrics
    if valid_examples == 0:
        return {'f1': 0.0, 'em': 0.0, 'valid_examples': 0}
    
    return {
        'f1': total_f1 / valid_examples,
        'em': total_em / valid_examples,
        'valid_examples': valid_examples
    }

def tune_hyperparameters(val_file, model_path, model_name):
    """
    Run grid search over hyperparameters for a specific model
    Args:
        val_file: Path to validation data JSON
        model_path: Path to QA model directory
        model_name: Name of the model (for logging)
    Returns:
        dict: Best hyperparameter configuration"""
    
    print("="*60)
    print(f"MODULE 6: HYPERPARAMETER TUNING FOR {model_name}")
    print("="*60)
    
    # Load validation data
    print(f"\n[1] Loading validation data: {val_file}")
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    print(f" Loaded {len(val_data)} validation examples")
    
    # Initialize retriever and QA model
    print("\n[2] Initializing models...")
    print("    Loading retriever...")

    retriever = HybridRetriever(index_dir="data/indices")
    retriever.load_indices()
    print(" Retriever loaded")
    
    print(f"    Loading QA model ({model_name})...")
    qa_model = QAModel(model_path, device="mps")
    print(" QA model loaded")
    
    # Define hyperparameter grid
    param_grid = {
        'top_k': [5, 10, 15],           # Retrieval: passages to retrieve
        'rrf_k': [30, 60, 90],          # Retrieval: RRF fusion parameter
        'qa_top_k': [3, 5, 10]          # QA: passages to process
    }
    
    print(f"\n[3] Hyperparameter grid:")
    for param, values in param_grid.items():
        print(f"    {param:12s}: {values}")
    
    total_configs = (len(param_grid['top_k']) * 
                     len(param_grid['rrf_k']) * 
                     len(param_grid['qa_top_k']))
    print(f"\n    Total configurations: {total_configs}")
    
    # Grid search
    print(f"\n[4] Running grid search...")
    results = []
    
    configs = list(product(
        param_grid['top_k'],
        param_grid['rrf_k'],
        param_grid['qa_top_k']
    ))
    
    start_time = time.time()
    
    for top_k, rrf_k, qa_top_k in tqdm(configs, desc="    Evaluating"):
        metrics = evaluate_configuration(
            retriever, qa_model, val_data,
            top_k, rrf_k, qa_top_k
        )
        
        results.append({
            'top_k': top_k,
            'rrf_k': rrf_k,
            'qa_top_k': qa_top_k,
            'f1': metrics['f1'],
            'em': metrics['em'],
            'valid_examples': metrics['valid_examples']
        })
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    df = pd.DataFrame(results)
    
    # Find best configuration
    best_idx = df['f1'].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n[5] Grid search complete!")
    print(f"    Time elapsed: {elapsed_time:.1f}s")
    print(f"    Time per config: {elapsed_time/total_configs:.1f}s")
    
    # Print best configuration
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION FOR {model_name} (highest F1 on validation)")
    print("="*60)
    print(f"  top_k:          {int(best['top_k'])}")
    print(f"  rrf_k:          {int(best['rrf_k'])}")
    print(f"  qa_top_k:       {int(best['qa_top_k'])}")
    print(f"\n  Validation F1:  {best['f1']:.4f}")
    print(f"  Validation EM:  {best['em']:.4f}")
    print(f"  Valid examples: {int(best['valid_examples'])}/{len(val_data)}")
    
    # Show top 10 configurations
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    top10 = df.nlargest(10, 'f1')
    print(top10[['top_k', 'rrf_k', 'qa_top_k', 'f1', 'em']].to_string(index=False))
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save full grid search results
    csv_file = output_dir / f"hyperparameter_grid_search_{model_name.lower()}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n[6] Full results saved to: {csv_file}")
    
    # Save best configuration
    best_config = {
        'model': model_name,
        'top_k': int(best['top_k']),
        'rrf_k': int(best['rrf_k']),
        'qa_top_k': int(best['qa_top_k']),
        'validation_f1': float(best['f1']),
        'validation_em': float(best['em']),
        'tuned_on_examples': len(val_data),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_file = output_dir / f"best_hyperparameters_{model_name.lower()}.json"
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"    Best config saved to: {config_file}")
    
    print(f"\n HYPERPARAMETER TUNING COMPLETE FOR {model_name}")
    print("="*60)
    
    return best_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bioasq", help="pubmedqa or bioasq")
    parser.add_argument("--model", type=str, default="roberta", 
                       choices=["roberta", "scibert", "both"],
                       help="Which model to tune hyperparameters for")
    args = parser.parse_args()

    # Choose validation file based on dataset
    if args.dataset == "bioasq":
        val_file = "data/processed/bioasq_val.json"
    else:
        val_file = "data/processed/val_extractive.json"

    # Tune for specified model(s)
    if args.model in ["roberta", "both"]:
        print("\n" + "="*30)
        print("TUNING ROBERTA")
        print("="*30)
        roberta_config = tune_hyperparameters(
            val_file, 
            "models/roberta_bioasq/",
            "RoBERTa"
        )
    
    if args.model in ["scibert", "both"]:
        print("\n" + "="*30)
        print("TUNING SCIBERT")
        print("="*30)
        scibert_config = tune_hyperparameters(
            val_file,
            "models/scibert_bioasq/",
            "SciBERT"
        )
    
    print("\n" + "="*30)
    print("ALL HYPERPARAMETER TUNING COMPLETE")
    print("="*30)
