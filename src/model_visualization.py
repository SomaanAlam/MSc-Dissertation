import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

# Load evaluation and statistical data
with open("results/evaluation_results_20251211_025455.json") as f:
    results = json.load(f)
with open("results/statistical_analysis.json") as f:
    stats = json.load(f)

rob_df = pd.DataFrame(results['roberta'])
sci_df = pd.DataFrame(results['scibert'])
rob_df['model'] = 'RoBERTa'
sci_df['model'] = 'SciBERT'
df = pd.concat([rob_df, sci_df])

colors = {'RoBERTa': '#3498db', 'SciBERT': '#e74c3c'}

# --- FIGURE 1 --- F1 Comparison
fig1, ax1 = plt.subplots(figsize=(6, 5))
means = [results['roberta_summary']['f1_mean'], results['scibert_summary']['f1_mean']]
stds = [results['roberta_summary']['f1_std'], results['scibert_summary']['f1_std']]
labels = ['RoBERTa', 'SciBERT']
bar = ax1.bar(labels, means, yerr=stds, capsize=10, color=[colors[l] for l in labels], edgecolor='black')
for i, val in enumerate(means):
    ax1.text(i, val + 0.02, f"{val:.4f}", ha='center', fontweight='bold')
ax1.set_title("Model Performance Comparison (F1 Score)", fontweight='bold')
ax1.set_ylabel("F1 Score", fontweight='bold')
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
fig1.tight_layout()
fig1.savefig("results/figures/fig1_f1_comparison.png")

# --- FIGURE 2 --- Violin plot only
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.violinplot(data=df, x='model', y='f1', palette=colors, ax=ax2)
ax2.set_title("F1 Score Distribution (Violin Plot)", fontweight='bold')
ax2.set_ylabel("F1 Score", fontweight='bold')
ax2.set_xlabel("")
ax2.grid(axis='y', alpha=0.3, linestyle='--')
fig2.tight_layout()
fig2.savefig("results/figures/fig2_f1_distributions.png")

# --- FIGURE 3 --- Scatter Comparison
fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.scatter(rob_df['f1'], sci_df['f1'], alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
max_f1 = max(df['f1'].max(), 1)
ax3.plot([0, max_f1], [0, max_f1], 'k--', alpha=0.5)
ax3.set_title("Per-Example F1 Score Comparison", fontweight='bold')
ax3.set_xlabel("RoBERTa F1 Score", fontweight='bold')
ax3.set_ylabel("SciBERT F1 Score", fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.3)
fig3.tight_layout()
fig3.savefig("results/figures/fig3_scatter_comparison.png")

# --- FIGURE 4 --- Latency Comparison + Breakdown
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 5))
means = [results['roberta_summary']['total_latency_mean_ms'], results['scibert_summary']['total_latency_mean_ms']]
stds = [results['roberta_summary']['qa_latency_std_ms'], results['scibert_summary']['qa_latency_std_ms']]
bar = ax4a.bar(labels, means, yerr=stds, capsize=10, color=[colors[l] for l in labels], edgecolor='black')
for i, val in enumerate(means):
    ax4a.text(i, val + 10, f"{val:.1f} ms", ha='center', fontweight='bold')
ax4a.set_title("Average Total Latency", fontweight='bold')
ax4a.set_ylabel("Latency (ms)", fontweight='bold')
ax4a.grid(axis='y', alpha=0.3, linestyle='--')

retrieval = [results['roberta_summary']['retrieval_latency_mean_ms'], results['scibert_summary']['retrieval_latency_mean_ms']]
qa = [results['roberta_summary']['qa_latency_mean_ms'], results['scibert_summary']['qa_latency_mean_ms']]
x = range(2)
ax4b.bar(x, retrieval, label='Retrieval', color='skyblue', edgecolor='black')
ax4b.bar(x, qa, bottom=retrieval, label='QA', color=[colors['RoBERTa'], colors['SciBERT']], edgecolor='black', alpha=0.7)
ax4b.set_xticks(x)
ax4b.set_xticklabels(labels)
ax4b.set_title("Latency Breakdown", fontweight='bold')
ax4b.set_ylabel("Latency (ms)", fontweight='bold')
ax4b.legend()
ax4b.grid(axis='y', alpha=0.3, linestyle='--')
fig4.tight_layout()
fig4.savefig("results/figures/fig4_latency_comparison.png")
