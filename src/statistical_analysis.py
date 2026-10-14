"""
Module 9: Statistical Analysis
Implements rigorous statistical testing for model comparison and evaluation
Includes paired t-tests, effect sizes, McNemar's test, bootstrap CIs, correlation analysis
Latency comparisons between RoBERTa and SciBERT QA models

"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import pandas as pd

class StatisticalAnalyzer:
    def __init__(self, results_file: str):
        """
        Initialize with evaluation results
        
        Args:
            results_file: Path to evaluation_results_*.json
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.roberta_data = self.results['roberta']
        self.scibert_data = self.results['scibert']
        
        print(f" Loaded results: {len(self.roberta_data)} examples")
    
    def bootstrap_confidence_interval(self, 
                                     data: List[float], 
                                     n_iterations: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval
        
        Args:
            data: List of metric values
            n_iterations: Number of bootstrap samples
            confidence_level: CI level (default 0.95)
            
        Returns:
            (lower_bound, upper_bound)
        """
        np.random.seed(42)  # Reproducibility
        
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_iterations):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def paired_t_test(self, 
                     metric: str = 'f1',
                     alternative: str = 'two-sided') -> Dict:
        """
        Paired t-test comparing RoBERTa vs SciBERT
        
        Args:
            metric: Metric to compare ('f1', 'exact_match', 'confidence')
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results
        """
        roberta_scores = [ex[metric] for ex in self.roberta_data]
        scibert_scores = [ex[metric] for ex in self.scibert_data]
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(scibert_scores, roberta_scores, 
                                                alternative=alternative)
        
        # Calculate means
        rob_mean = np.mean(roberta_scores)
        sci_mean = np.mean(scibert_scores)
        
        # Calculate difference
        differences = np.array(scibert_scores) - np.array(roberta_scores)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # 95% CI for difference
        ci_lower, ci_upper = stats.t.interval(
            0.95, 
            len(differences)-1,
            loc=mean_diff, 
            scale=stats.sem(differences)
        )
        
        return {
            'metric': metric,
            'roberta_mean': rob_mean,
            'scibert_mean': sci_mean,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            't_statistic': t_statistic,
            'p_value': p_value,
            'df': len(differences) - 1,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'alternative': alternative
        }
    
    def cohens_d(self, metric: str = 'f1') -> Dict:
        """
        Calculate Cohen's d effect size
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with effect size results
        """
        roberta_scores = np.array([ex[metric] for ex in self.roberta_data])
        scibert_scores = np.array([ex[metric] for ex in self.scibert_data])
        
        # Calculate paired differences
        differences = scibert_scores - roberta_scores
        
        # Cohen's d for paired samples
        d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'metric': metric,
            'cohens_d': d,
            'abs_cohens_d': abs_d,
            'interpretation': interpretation,
            'direction': 'SciBERT better' if d > 0 else 'RoBERTa better'
        }
    
    def mcnemar_test(self) -> Dict:
        """
        McNemar's test for dependent classifiers
        Tests if models disagree systematically
        
        Only applicable to binary outcomes (correct/incorrect)
        """
        # Create binary outcomes (F1 > 0 = correct)
        roberta_correct = [ex['f1'] > 0 for ex in self.roberta_data]
        scibert_correct = [ex['f1'] > 0 for ex in self.scibert_data]
        
        # Create contingency table
        both_correct = sum(r and s for r, s in zip(roberta_correct, scibert_correct))
        rob_only = sum(r and not s for r, s in zip(roberta_correct, scibert_correct))
        sci_only = sum(not r and s for r, s in zip(roberta_correct, scibert_correct))
        both_wrong = sum(not r and not s for r, s in zip(roberta_correct, scibert_correct))
        
        # McNemar statistic (with continuity correction)
        if rob_only + sci_only > 0:
            statistic = (abs(rob_only - sci_only) - 1) ** 2 / (rob_only + sci_only)
            p_value = stats.chi2.sf(statistic, 1)
        else:
            statistic = 0
            p_value = 1.0
        
        return {
            'contingency_table': {
                'both_correct': both_correct,
                'roberta_only_correct': rob_only,
                'scibert_only_correct': sci_only,
                'both_wrong': both_wrong
            },
            'statistic': statistic,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'model_agreement_rate': (both_correct + both_wrong) / len(roberta_correct)
        }
    
    def bootstrap_comparison(self, 
                            metric: str = 'f1',
                            n_iterations: int = 1000) -> Dict:
        """
        Bootstrap confidence intervals for both models
        
        Args:
            metric: Metric to analyze
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with CI for both models
        """
        roberta_scores = [ex[metric] for ex in self.roberta_data]
        scibert_scores = [ex[metric] for ex in self.scibert_data]
        
        rob_ci = self.bootstrap_confidence_interval(roberta_scores, n_iterations)
        sci_ci = self.bootstrap_confidence_interval(scibert_scores, n_iterations)
        
        rob_mean = np.mean(roberta_scores)
        sci_mean = np.mean(scibert_scores)
        
        return {
            'metric': metric,
            'roberta': {
                'mean': rob_mean,
                'ci_95_lower': rob_ci[0],
                'ci_95_upper': rob_ci[1],
                'ci_width': rob_ci[1] - rob_ci[0]
            },
            'scibert': {
                'mean': sci_mean,
                'ci_95_lower': sci_ci[0],
                'ci_95_upper': sci_ci[1],
                'ci_width': sci_ci[1] - sci_ci[0]
            },
            'cis_overlap': not (rob_ci[1] < sci_ci[0] or sci_ci[1] < rob_ci[0])
        }
    
    def correlation_analysis(self) -> Dict:
        """
        Analyze correlation between model predictions
        """
        # F1 scores
        rob_f1 = np.array([ex['f1'] for ex in self.roberta_data])
        sci_f1 = np.array([ex['f1'] for ex in self.scibert_data])
        
        # Confidence scores
        rob_conf = np.array([ex['confidence'] for ex in self.roberta_data])
        sci_conf = np.array([ex['confidence'] for ex in self.scibert_data])
        
        # Pearson correlation
        f1_corr, f1_p = stats.pearsonr(rob_f1, sci_f1)
        conf_corr, conf_p = stats.pearsonr(rob_conf, sci_conf)
        
        # Spearman correlation (rank-based)
        f1_spearman, f1_spearman_p = stats.spearmanr(rob_f1, sci_f1)
        
        return {
            'f1_pearson_r': f1_corr,
            'f1_pearson_p': f1_p,
            'f1_spearman_rho': f1_spearman,
            'f1_spearman_p': f1_spearman_p,
            'confidence_pearson_r': conf_corr,
            'confidence_pearson_p': conf_p,
            'interpretation': {
                'f1_correlation': 'strong' if abs(f1_corr) > 0.7 else 'moderate' if abs(f1_corr) > 0.4 else 'weak',
                'models_agree': f1_corr > 0.5
            }
        }
    
    def latency_comparison(self) -> Dict:
        """
        Statistical comparison of latencies
        """
        rob_latency = np.array([ex['total_latency'] * 1000 for ex in self.roberta_data])
        sci_latency = np.array([ex['total_latency'] * 1000 for ex in self.scibert_data])
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(sci_latency, rob_latency)
        
        # Descriptive stats
        rob_mean = np.mean(rob_latency)
        sci_mean = np.mean(sci_latency)
        difference = sci_mean - rob_mean
        pct_difference = (difference / rob_mean) * 100
        
        return {
            'roberta_mean_ms': rob_mean,
            'roberta_std_ms': np.std(rob_latency),
            'scibert_mean_ms': sci_mean,
            'scibert_std_ms': np.std(sci_latency),
            'difference_ms': difference,
            'pct_difference': pct_difference,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'faster_model': 'RoBERTa' if rob_mean < sci_mean else 'SciBERT'
        }
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete statistical analysis
        
        Returns:
            Dictionary with all analysis results
        """
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*60}\n")
        
        results = {}
        
        # 1. Paired t-test (F1)
        print("[1] Paired t-test (F1 Score)...")
        results['paired_t_test_f1'] = self.paired_t_test('f1')
        
        # 2. Paired t-test (EM)
        print("[2] Paired t-test (Exact Match)...")
        results['paired_t_test_em'] = self.paired_t_test('exact_match')
        
        # 3. Cohen's d (F1)
        print("[3] Effect size (Cohen's d)...")
        results['cohens_d_f1'] = self.cohens_d('f1')
        
        # 4. McNemar's test
        print("[4] McNemar's test...")
        results['mcnemar_test'] = self.mcnemar_test()
        
        # 5. Bootstrap CIs
        print("[5] Bootstrap confidence intervals...")
        results['bootstrap_f1'] = self.bootstrap_comparison('f1', n_iterations=1000)
        results['bootstrap_em'] = self.bootstrap_comparison('exact_match', n_iterations=1000)
        
        # 6. Correlation analysis
        print("[6] Correlation analysis...")
        results['correlation'] = self.correlation_analysis()
        
        # 7. Latency comparison
        print("[7] Latency comparison...")
        results['latency'] = self.latency_comparison()
        
        print("\n Statistical analysis complete")
        
        return results
    
    def print_report(self, results: Dict):
        """
        Print human-readable statistical report
        """
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS REPORT")
        print(f"{'='*60}\n")
        
        # Sample size
        print(f"Sample Size: {len(self.roberta_data)} examples\n")
        
        # 1. Paired t-test (F1)
        print(f"{'─'*60}")
        print("1. PAIRED T-TEST (F1 Score)")
        print(f"{'─'*60}")
        t_test = results['paired_t_test_f1']
        print(f"  RoBERTa mean:  {t_test['roberta_mean']:.4f}")
        print(f"  SciBERT mean:  {t_test['scibert_mean']:.4f}")
        print(f"  Difference:    {t_test['mean_difference']:.4f} ± {t_test['std_difference']:.4f}")
        print(f"  95% CI:        [{t_test['ci_95_lower']:.4f}, {t_test['ci_95_upper']:.4f}]")
        print(f"  t-statistic:   {t_test['t_statistic']:.3f}")
        print(f"  p-value:       {t_test['p_value']:.6f}")
        print(f"  Significant:   {' YES (p < 0.05)' if t_test['significant_at_0.05'] else ' NO'}")
        
        # 2. Effect size
        print(f"\n{'─'*60}")
        print("2. EFFECT SIZE (Cohen's d)")
        print(f"{'─'*60}")
        cohens = results['cohens_d_f1']
        print(f"  Cohen's d:     {cohens['cohens_d']:.3f}")
        print(f"  Interpretation: {cohens['interpretation'].upper()}")
        print(f"  Direction:     {cohens['direction']}")
        
        # 3. McNemar's test
        print(f"\n{'─'*60}")
        print("3. MCNEMAR'S TEST (Model Agreement)")
        print(f"{'─'*60}")
        mcnemar = results['mcnemar_test']
        ct = mcnemar['contingency_table']
        print(f"  Contingency Table:")
        print(f"                      SciBERT Correct | SciBERT Wrong")
        print(f"    RoBERTa Correct:       {ct['both_correct']:4d}      |     {ct['roberta_only_correct']:4d}")
        print(f"    RoBERTa Wrong:         {ct['scibert_only_correct']:4d}      |     {ct['both_wrong']:4d}")
        print(f"  ")
        print(f"  Statistic:     {mcnemar['statistic']:.3f}")
        print(f"  p-value:       {mcnemar['p_value']:.6f}")
        print(f"  Significant:   {' YES' if mcnemar['significant_at_0.05'] else ' NO'}")
        print(f"  Agreement rate: {mcnemar['model_agreement_rate']:.2%}")
        
        # 4. Bootstrap CIs
        print(f"\n{'─'*60}")
        print("4. BOOTSTRAP CONFIDENCE INTERVALS (F1)")
        print(f"{'─'*60}")
        bootstrap = results['bootstrap_f1']
        print(f"  RoBERTa:")
        print(f"    Mean:    {bootstrap['roberta']['mean']:.4f}")
        print(f"    95% CI:  [{bootstrap['roberta']['ci_95_lower']:.4f}, {bootstrap['roberta']['ci_95_upper']:.4f}]")
        print(f"  SciBERT:")
        print(f"    Mean:    {bootstrap['scibert']['mean']:.4f}")
        print(f"    95% CI:  [{bootstrap['scibert']['ci_95_lower']:.4f}, {bootstrap['scibert']['ci_95_upper']:.4f}]")
        print(f"  CIs Overlap: {'Yes' if bootstrap['cis_overlap'] else 'No'}")
        
        # 5. Correlation
        print(f"\n{'─'*60}")
        print("5. CORRELATION ANALYSIS")
        print(f"{'─'*60}")
        corr = results['correlation']
        print(f"  F1 Pearson r:  {corr['f1_pearson_r']:.3f} (p={corr['f1_pearson_p']:.6f})")
        print(f"  F1 Spearman ρ: {corr['f1_spearman_rho']:.3f} (p={corr['f1_spearman_p']:.6f})")
        print(f"  Interpretation: {corr['interpretation']['f1_correlation'].upper()} correlation")
        print(f"  Models agree:   {' YES' if corr['interpretation']['models_agree'] else 'NO'}")
        
        # 6. Latency
        print(f"\n{'─'*60}")
        print("6. LATENCY COMPARISON")
        print(f"{'─'*60}")
        lat = results['latency']
        print(f"  RoBERTa:       {lat['roberta_mean_ms']:.1f} ± {lat['roberta_std_ms']:.1f} ms")
        print(f"  SciBERT:       {lat['scibert_mean_ms']:.1f} ± {lat['scibert_std_ms']:.1f} ms")
        print(f"  Difference:    {lat['difference_ms']:+.1f} ms ({lat['pct_difference']:+.1f}%)")
        print(f"  Significant:   {'YES' if lat['significant'] else ' NO'}")
        print(f"  Faster model:  {lat['faster_model']}")
        
 
    def save_results(self, results: Dict, output_dir: str = "results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        filename = output_dir / "statistical_analysis.json"
        
        # Use a custom converter for NumPy types
        def convert(o):
            if isinstance(o, (np.integer, np.int64)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            if isinstance(o, (np.bool_)): return bool(o)
            return str(o)  # fallback for anything else

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=convert)

        print(f" Statistical analysis saved to: {filename}")
        return filename


def find_latest_results(results_dir: str = "results") -> str:
    """Find the most recent evaluation results file"""
    results_path = Path(results_dir)
    
    # Find all evaluation result files
    result_files = list(results_path.glob("evaluation_results_*.json"))
    
    if not result_files:
        raise FileNotFoundError(f"No evaluation results found in {results_dir}")
    
    # Sort by modification time, get most recent
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    return str(latest_file)


def main():
    """Main execution"""
    print("="*60)
    print("MODULE 7: STATISTICAL ANALYSIS")
    print("="*60)
    
    # Find latest results
    try:
        results_file = find_latest_results()
        print(f"\n Found results: {results_file}")
    except FileNotFoundError as e:
        print(f"\n {e}")
        print("\nPlease run evaluation first: python src/evaluator.py")
        return
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(results_file)
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    # Print report
    analyzer.print_report(results)
    
    # Save results
    analyzer.save_results(results)
    
    print("\n Statistical analysis complete!")


if __name__ == "__main__":
    main()
