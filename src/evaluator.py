"""
Module 8: Full Evaluation Pipeline for RoBERTa and SciBERT QA Models
Combines retrieval and QA with model-specific hyperparameters
Evaluates on test set and saves results for analysis

"""

import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import re
import argparse

from src.retriever import HybridRetriever
from src.qa_models import QAModel

class QAEvaluator:
    """QA Evaluator for RoBERTa and SciBERT models
    """
    def __init__(self):
        self.results = {
            'roberta': [],
            'scibert': [],
            'metadata': {},
            'config': {}
        }
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer for comparison (SQuAD-style)
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth
        """
        pred_tokens = QAEvaluator.normalize_answer(prediction).split()
        truth_tokens = QAEvaluator.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 and len(truth_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def compute_exact_match(prediction: str, ground_truth: str) -> float:
        """Compute exact match  between prediction and ground truth
        """
        return float(QAEvaluator.normalize_answer(prediction) == 
                    QAEvaluator.normalize_answer(ground_truth))
    
    def load_hyperparameters(self, model_name: str) -> Dict:
        """Load model-specific hyperparameters
        
        Args:
            model_name: 'roberta' or 'scibert'
        Returns:
            Dict of hyperparameters"""
        config_file = f"results/best_hyperparameters_{model_name.lower()}.json"
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"  No hyperparameter config found for {model_name} at {config_file}")
            print(f"    Using default hyperparameters")
            return {
                'top_k': 10,
                'rrf_k': 60,
                'qa_top_k': 5
            }
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f" Loaded hyperparameters for {model_name}:")
        print(f"    top_k:    {config['top_k']}")
        print(f"    rrf_k:    {config['rrf_k']}")
        print(f"    qa_top_k: {config['qa_top_k']}")
        print(f"    Val F1:   {config['validation_f1']:.4f}")
        
        return config
    
    def run_full_evaluation(self,
                           test_data: List[Dict],
                           max_examples: int = None) -> Dict:
        """
        Run full evaluation on test set with both models using their optimal configs
        Args:
            test_data: List of test examples
            max_examples: Optional limit on number of examples to evaluate
        Returns:
            Dict of results
        """
        if max_examples:
            test_data = test_data[:max_examples]
        
        print(f"\n{'='*60}")
        print(f"FULL EVALUATION PIPELINE (Model-Specific Hyperparameters)")
        print(f"{'='*60}\n")
        
        # Load each model hyperparameters
        roberta_config = self.load_hyperparameters('roberta')
        scibert_config = self.load_hyperparameters('scibert')
        
        self.results['config'] = {
            'roberta': roberta_config,
            'scibert': scibert_config
        }
        
        # Initialize QA models
        print(f"\n[1] Loading QA models...")
        print(f"    Loading RoBERTa...")
        roberta = QAModel("models/roberta_bioasq/", device="mps")
        print(f"    Loading SciBERT...")
        scibert = QAModel("models/scibert_bioasq/", device="mps")
        print(f" Both models loaded")
        
        # Run evaluation
        print(f"\n[2] Running evaluation on {len(test_data)} examples...\n")
        
        # Track timing
        roberta_retrieval_times = []
        roberta_qa_times = []
        scibert_retrieval_times = []
        scibert_qa_times = []
        
        retriever_rob = HybridRetriever(rrf_k=roberta_config['rrf_k'], device="mps")
        retriever_rob.load_indices()

        retriever_sci = HybridRetriever(rrf_k=scibert_config['rrf_k'], device="cpu")
        retriever_sci.load_indices()

        for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
            question = example['question']
            ground_truth = example['answer']
            
            # ROBERTA EVALUATION (with RoBERTa-optimized hyperparameters)
                        
            start_time = time.time()
            passages_rob = retriever_rob.search(question, top_k=roberta_config['top_k'])
            roberta_retrieval_times.append(time.time() - start_time)
            
            start_time = time.time()
            rob_pred = roberta.answer_with_passages(
                question, 
                passages_rob, 
                top_k=roberta_config['qa_top_k']
            )
            roberta_qa_times.append(time.time() - start_time)
            
            # Evaluate RoBERTa
            rob_f1 = self.compute_f1(rob_pred['answer'], ground_truth)
            rob_em = self.compute_exact_match(rob_pred['answer'], ground_truth)
            
            self.results['roberta'].append({
                'id': example.get('id', f'example_{i}'),
                'question': question,
                'prediction': rob_pred['answer'],
                'ground_truth': ground_truth,
                'confidence': rob_pred['confidence'],
                'combined_score': rob_pred.get('combined_score', rob_pred['confidence']),
                'no_answer': rob_pred.get('no_answer', False),
                'f1': rob_f1,
                'exact_match': rob_em,
                'retrieval_latency': roberta_retrieval_times[-1],
                'qa_latency': roberta_qa_times[-1],
                'total_latency': roberta_retrieval_times[-1] + roberta_qa_times[-1]
            })
            
            # SCIBERT EVALUATION (with SciBERT-optimized hyperparameters)
            
            start_time = time.time()
            passages_sci = retriever_sci.search(question, top_k=scibert_config['top_k'])
            scibert_retrieval_times.append(time.time() - start_time)
            
            start_time = time.time()
            sci_pred = scibert.answer_with_passages(
                question, 
                passages_sci, 
                top_k=scibert_config['qa_top_k']
            )
            scibert_qa_times.append(time.time() - start_time)
            
            # Evaluate SciBERT
            sci_f1 = self.compute_f1(sci_pred['answer'], ground_truth)
            sci_em = self.compute_exact_match(sci_pred['answer'], ground_truth)
            
            self.results['scibert'].append({
                'id': example.get('id', f'example_{i}'),
                'question': question,
                'prediction': sci_pred['answer'],
                'ground_truth': ground_truth,
                'confidence': sci_pred['confidence'],
                'combined_score': sci_pred.get('combined_score', sci_pred['confidence']),
                'no_answer': sci_pred.get('no_answer', False),
                'f1': sci_f1,
                'exact_match': sci_em,
                'retrieval_latency': scibert_retrieval_times[-1],
                'qa_latency': scibert_qa_times[-1],
                'total_latency': scibert_retrieval_times[-1] + scibert_qa_times[-1]
            })
        
        # Calculate summary statistics
        self.results['metadata'] = {
            'num_examples': len(test_data),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'roberta_model': 'models/roberta_bioasq/',
            'scibert_model': 'models/scibert_bioasq/',
        }
        
        self._calculate_summary_stats()
        
        return self.results
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics for both models
        Populates self.results with summary statistics
        """
        for model in ['roberta', 'scibert']:
            results = self.results[model]
            
            f1_scores = [r['f1'] for r in results]
            em_scores = [r['exact_match'] for r in results]
            confidences = [r['confidence'] for r in results]
            combined_scores = [r['combined_score'] for r in results]
            retrieval_latencies = [r['retrieval_latency'] for r in results]
            qa_latencies = [r['qa_latency'] for r in results]
            total_latencies = [r['total_latency'] for r in results]
            no_answer_count = sum(r['no_answer'] for r in results)
            
            self.results[f'{model}_summary'] = {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'f1_median': np.median(f1_scores),
                'f1_min': np.min(f1_scores),
                'f1_max': np.max(f1_scores),
                'exact_match_mean': np.mean(em_scores),
                'exact_match_std': np.std(em_scores),
                'confidence_mean': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'combined_score_mean': np.mean(combined_scores),
                'combined_score_std': np.std(combined_scores),
                'retrieval_latency_mean_ms': np.mean(retrieval_latencies) * 1000,
                'retrieval_latency_std_ms': np.std(retrieval_latencies) * 1000,
                'qa_latency_mean_ms': np.mean(qa_latencies) * 1000,
                'qa_latency_std_ms': np.std(qa_latencies) * 1000,
                'total_latency_mean_ms': np.mean(total_latencies) * 1000,
                'total_latency_p95_ms': np.percentile(total_latencies, 95) * 1000,
                'total_latency_p99_ms': np.percentile(total_latencies, 99) * 1000,
                'no_answer_rate': no_answer_count / len(results),
                'num_examples': len(results)
            }
    
    def print_summary(self):
        """Print summary results to console
        """
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Dataset: {self.results['metadata']['num_examples']} test examples")
        print(f"Date: {self.results['metadata']['evaluation_date']}")
        
        for model_name in ['roberta', 'scibert']:
            config = self.results['config'][model_name]
            summary = self.results[f'{model_name}_summary']
            
            print(f"\n{'─'*60}")
            print(f"{model_name.upper()} Results:")
            print(f"{'─'*60}")
            print(f"  Hyperparameters Used:")
            print(f"    top_k:    {config['top_k']}")
            print(f"    rrf_k:    {config['rrf_k']}")
            print(f"    qa_top_k: {config['qa_top_k']}")
            print(f"\n  Answer Quality:")
            print(f"    F1 Score:      {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
            print(f"    F1 Range:      [{summary['f1_min']:.4f}, {summary['f1_max']:.4f}]")
            print(f"    Exact Match:   {summary['exact_match_mean']:.4f} ± {summary['exact_match_std']:.4f}")
            print(f"    No Answer Rate: {summary['no_answer_rate']:.2%}")
        
        # Comparison
        rob_f1 = self.results['roberta_summary']['f1_mean']
        sci_f1 = self.results['scibert_summary']['f1_mean']
        improvement = ((sci_f1 - rob_f1) / rob_f1) * 100 if rob_f1 > 0 else 0
        
        rob_em = self.results['roberta_summary']['exact_match_mean']
        sci_em = self.results['scibert_summary']['exact_match_mean']
        em_improvement = ((sci_em - rob_em) / rob_em) * 100 if rob_em > 0 else 0
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"F1 Score:")
        print(f"  RoBERTa:     {rob_f1:.4f}")
        print(f"  SciBERT:     {sci_f1:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Winner:      {'SciBERT' if sci_f1 > rob_f1 else 'RoBERTa'}")
        
        print(f"\nExact Match:")
        print(f"  RoBERTa:     {rob_em:.4f}")
        print(f"  SciBERT:     {sci_em:.4f}")
        if rob_em > 0:
            print(f"  Improvement: {em_improvement:+.2f}%")
        
        print(f"\n{'='*60}")
    
    def save_results(self, output_dir: str = "results"):
        """Save results to JSON file
        
        Args:
            output_dir: Directory to save results
        Returns:
            Path to saved JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n Results saved to: {filename}")
        return filename
    
    def export_for_analysis(self, output_dir: str = "results"):
        """Export results in analysis-friendly format
        Args:
            output_dir: Directory to save CSV
        Returns:
            Path to saved CSV file"""
        import pandas as pd
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create DataFrames
        rob_df = pd.DataFrame(self.results['roberta'])
        sci_df = pd.DataFrame(self.results['scibert'])
        
        # Add model column
        rob_df['model'] = 'RoBERTa'
        sci_df['model'] = 'SciBERT'
        
        # Combine
        combined_df = pd.concat([rob_df, sci_df], ignore_index=True)
        
        # Save CSV
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_file = output_dir / f"evaluation_data_{timestamp}.csv"
        combined_df.to_csv(csv_file, index=False)
        
        print(f" CSV exported to: {csv_file}")
        return csv_file


def main():
    """Main execution"""
    print("="*60)
    print("MODULE: FULL EVALUATION")
    print("="*60)
    
    # Load test data
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bioasq", help="Dataset name (pubmedqa or bioasq)")
    args = parser.parse_args()

    if args.dataset == "bioasq":
        test_file = "data/processed/bioasq_test.json"
    else:
        test_file = "data/processed/test_extractive.json"
        
    print(f"\n Loading test data: {test_file}")
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f" Loaded {len(test_data)} test examples")
    
    # Ask for confirmation
    print(f"\nThis will evaluate both RoBERTa and SciBERT on {len(test_data)} examples.")
    print(f"Each model will use its own optimized hyperparameters.")
    print(f"Estimated time: ~{len(test_data) * 0.5 / 60:.0f} minutes")
    
    response = input("\nProceed with full evaluation? (y/n): ")
    
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        return
    
    # Run evaluation
    evaluator = QAEvaluator()
    
    start_time = time.time()
    results = evaluator.run_full_evaluation(test_data)
    elapsed_time = time.time() - start_time
    
    # Print results
    evaluator.print_summary()
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    json_file = evaluator.save_results()
    csv_file = evaluator.export_for_analysis()
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per example: {elapsed_time/len(test_data):.2f} seconds")

if __name__ == "__main__":
    main()