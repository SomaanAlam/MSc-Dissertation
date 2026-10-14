"""
Dataset Visualization
Generates EDA figures for BioASQ dataset analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def load_dataset(filepath):
    """Load dataset from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_dataset(data, split_name):
    """Extract statistics from dataset"""
    stats = {
        'split': split_name,
        'n_examples': len(data),
        'context_lengths': [len(ex['context']) for ex in data],
        'question_lengths': [len(ex['question']) for ex in data],
        'answer_lengths': [len(ex['answer']) for ex in data],
        'answer_positions': [],
        'token_estimates': []
    }
    
    for ex in data:
        # Answer position (relative)
        pos = ex['context'].lower().find(ex['answer'].lower())
        if pos != -1:
            relative_pos = pos / len(ex['context'])
            stats['answer_positions'].append(relative_pos)
        
        # Token estimate (approx 4 chars per token)
        stats['token_estimates'].append(len(ex['context']) / 4)
    
    return stats


def plot_context_length_distribution(all_stats, output_dir):
    """Figure: Distribution of context lengths across splits"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    colors = {'train': '#3498db', 'val': '#e74c3c', 'test': '#2ecc71'}
    
    for idx, stats in enumerate(all_stats):
        ax = axes[idx]
        split = stats['split']
        
        ax.hist(stats['context_lengths'], bins=30, color=colors[split], 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(stats['context_lengths']), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["context_lengths"]):.0f}')
        ax.axvline(np.median(stats['context_lengths']), color='orange', 
                   linestyle=':', linewidth=2, label=f'Median: {np.median(stats["context_lengths"]):.0f}')
        
        ax.set_xlabel('Context Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{split.capitalize()} Set (n={stats["n_examples"]})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Distribution of Context Lengths Across Dataset Splits', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_context_length_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_context_length_distribution.png")


def plot_answer_length_distribution(all_stats, output_dir):
    """Figure: Distribution of answer lengths"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'train': '#3498db', 'val': '#e74c3c', 'test': '#2ecc71'}
    
    for stats in all_stats:
        split = stats['split']
        ax.hist(stats['answer_lengths'], bins=50, alpha=0.5, 
                label=f'{split.capitalize()} (mean={np.mean(stats["answer_lengths"]):.1f})',
                color=colors[split], edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Answer Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Answer Lengths Across Dataset Splits', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add annotation for short answers
    short_answers = sum(1 for stats in all_stats for a in stats['answer_lengths'] if a < 20)
    total = sum(stats['n_examples'] for stats in all_stats)
    ax.text(0.95, 0.95, f'Short answers (<20 chars): {short_answers/total:.1%}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_answer_length_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_answer_length_distribution.png")


def plot_answer_position_distribution(all_stats, output_dir):
    """Figure: Where in the context do answers appear?"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine all positions
    all_positions = []
    for stats in all_stats:
        all_positions.extend(stats['answer_positions'])
    
    ax.hist(all_positions, bins=20, color='#9b59b6', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    
    # Add region annotations
    start_pct = sum(1 for p in all_positions if p < 0.2) / len(all_positions)
    middle_pct = sum(1 for p in all_positions if 0.2 <= p <= 0.8) / len(all_positions)
    end_pct = sum(1 for p in all_positions if p > 0.8) / len(all_positions)
    
    ax.axvspan(0, 0.2, alpha=0.1, color='green', label=f'Start (0-20%): {start_pct:.1%}')
    ax.axvspan(0.2, 0.8, alpha=0.1, color='blue', label=f'Middle (20-80%): {middle_pct:.1%}')
    ax.axvspan(0.8, 1.0, alpha=0.1, color='red', label=f'End (80-100%): {end_pct:.1%}')
    
    ax.set_xlabel('Relative Position in Context (0 = start, 1 = end)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Answer Positions Within Context', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_answer_position_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_answer_position_distribution.png")


def plot_token_estimate_distribution(all_stats, output_dir):
    """Figure: Estimated token counts (important for model max_length)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_tokens = []
    for stats in all_stats:
        all_tokens.extend(stats['token_estimates'])
    
    ax.hist(all_tokens, bins=30, color='#1abc9c', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax.axvline(384, color='orange', linestyle='--', linewidth=2, 
               label='384 tokens (common limit)')
    ax.axvline(512, color='red', linestyle='--', linewidth=2, 
               label='512 tokens (BERT max)')
    
    # Calculate percentages
    over_384 = sum(1 for t in all_tokens if t > 384) / len(all_tokens)
    over_512 = sum(1 for t in all_tokens if t > 512) / len(all_tokens)
    
    ax.text(0.95, 0.95, f'>384 tokens: {over_384:.1%}\n>512 tokens: {over_512:.1%}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('Estimated Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Estimated Token Distribution (Context Only)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_token_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_token_distribution.png")


def plot_split_comparison(all_stats, output_dir):
    """Figure: Comparison of statistics across splits"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    splits = [s['split'] for s in all_stats]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Sample sizes
    ax = axes[0, 0]
    sizes = [s['n_examples'] for s in all_stats]
    bars = ax.bar(splits, sizes, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Number of Examples')
    ax.set_title('(A) Dataset Split Sizes')
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(size), ha='center', va='bottom', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 2. Mean context lengths
    ax = axes[0, 1]
    means = [np.mean(s['context_lengths']) for s in all_stats]
    stds = [np.std(s['context_lengths']) for s in all_stats]
    bars = ax.bar(splits, means, yerr=stds, color=colors, edgecolor='black', 
                  linewidth=1, capsize=5)
    ax.set_ylabel('Context Length (chars)')
    ax.set_title('(B) Mean Context Length (±1 SD)')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Mean answer lengths
    ax = axes[1, 0]
    means = [np.mean(s['answer_lengths']) for s in all_stats]
    stds = [np.std(s['answer_lengths']) for s in all_stats]
    bars = ax.bar(splits, means, yerr=stds, color=colors, edgecolor='black', 
                  linewidth=1, capsize=5)
    ax.set_ylabel('Answer Length (chars)')
    ax.set_title('(C) Mean Answer Length (±1 SD)')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Mean answer position
    ax = axes[1, 1]
    means = [np.mean(s['answer_positions']) for s in all_stats]
    stds = [np.std(s['answer_positions']) for s in all_stats]
    bars = ax.bar(splits, means, yerr=stds, color=colors, edgecolor='black', 
                  linewidth=1, capsize=5)
    ax.set_ylabel('Relative Position (0-1)')
    ax.set_title('(D) Mean Answer Position (±1 SD)')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Dataset Split Comparison', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_split_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_split_comparison.png")


def generate_statistics_table(all_stats, output_dir):
    """Generate LaTeX table for dissertation"""
    
    table_content = """
\\begin{table}[htbp]
\\centering
\\caption{BioASQ Dataset Statistics Across Splits}
\\label{tab:dataset-statistics}
\\begin{tabular}{lccc}
\\hline
\\textbf{Statistic} & \\textbf{Train} & \\textbf{Validation} & \\textbf{Test} \\\\
\\hline
"""
    
    for stats in all_stats:
        split = stats['split']
        if split == 'train':
            train = stats
        elif split == 'val':
            val = stats
        else:
            test = stats
    
    # Add rows
    table_content += f"Number of examples & {train['n_examples']} & {val['n_examples']} & {test['n_examples']} \\\\\n"
    table_content += f"Mean context length (chars) & {np.mean(train['context_lengths']):.1f} & {np.mean(val['context_lengths']):.1f} & {np.mean(test['context_lengths']):.1f} \\\\\n"
    table_content += f"Std context length & {np.std(train['context_lengths']):.1f} & {np.std(val['context_lengths']):.1f} & {np.std(test['context_lengths']):.1f} \\\\\n"
    table_content += f"Mean answer length (chars) & {np.mean(train['answer_lengths']):.1f} & {np.mean(val['answer_lengths']):.1f} & {np.mean(test['answer_lengths']):.1f} \\\\\n"
    table_content += f"Std answer length & {np.std(train['answer_lengths']):.1f} & {np.std(val['answer_lengths']):.1f} & {np.std(test['answer_lengths']):.1f} \\\\\n"
    table_content += f"Mean question length (chars) & {np.mean(train['question_lengths']):.1f} & {np.mean(val['question_lengths']):.1f} & {np.mean(test['question_lengths']):.1f} \\\\\n"
    table_content += f"Est. tokens >512 (\\%) & {sum(1 for t in train['token_estimates'] if t > 512)/len(train['token_estimates'])*100:.1f} & {sum(1 for t in val['token_estimates'] if t > 512)/len(val['token_estimates'])*100:.1f} & {sum(1 for t in test['token_estimates'] if t > 512)/len(test['token_estimates'])*100:.1f} \\\\\n"
    
    table_content += """\\hline
\\end{tabular}
\\end{table}
"""
    
    with open(output_dir / 'table_dataset_statistics.tex', 'w') as f:
        f.write(table_content)
    
    print(f"  Saved: table_dataset_statistics.tex")
    return table_content


def main():
    """Main execution"""
    print("="*60)
    print("DATA VISUALIZATION FOR CHAPTER 4")
    print("="*60)
    
    # Paths
    data_dir = Path("data/processed")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\n[1] Loading datasets...")
    train_data = load_dataset(data_dir / "bioasq_train.json")
    val_data = load_dataset(data_dir / "bioasq_val.json")
    test_data = load_dataset(data_dir / "bioasq_test.json")
    
    print(f"    Train: {len(train_data)} examples")
    print(f"    Val:   {len(val_data)} examples")
    print(f"    Test:  {len(test_data)} examples")
    
    # Analyze
    print("\n[2] Analyzing datasets...")
    train_stats = analyze_dataset(train_data, 'train')
    val_stats = analyze_dataset(val_data, 'val')
    test_stats = analyze_dataset(test_data, 'test')
    all_stats = [train_stats, val_stats, test_stats]
    
    # Generate figures
    print("\n[3] Generating figures...")
    plot_context_length_distribution(all_stats, output_dir)
    plot_answer_length_distribution(all_stats, output_dir)
    plot_answer_position_distribution(all_stats, output_dir)
    plot_token_estimate_distribution(all_stats, output_dir)
    plot_split_comparison(all_stats, output_dir)
    
    # Generate table
    print("\n[4] Generating LaTeX table...")
    table = generate_statistics_table(all_stats, output_dir)
    print(table)
    
    print("\n" + "="*60)
    print("DATA VISUALIZATION COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()