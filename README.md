# Domain Adaptation in Scientific Question Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MSc Dissertation - University of Essex**  
**Author:** Somaan Alam (2411759)  
**Supervisor:** Dr. Daniel F. Ahelegbey  
**Module:** MA981-7-AU

## Overview

This repository contains the complete implementation for investigating whether domain-specific pre-training (SciBERT) improves extractive question answering performance on biomedical literature compared to general-purpose models (RoBERTa), when integrated into a hybrid retrieval-augmented system.

## Project Structure

```
├── data/
│   ├── raw/                      # Raw BioASQ data
│   │   └── BioASQ/
│   │       └── training13b.json
│   ├── processed/                # Preprocessed splits
│   │   ├── bioasq_train.json
│   │   ├── bioasq_val.json
│   │   └── bioasq_test.json
│   └── indices/                  # BM25 and FAISS indices
│       ├── bm25_index.pkl
│       ├── faiss_index.bin
│       ├── embeddings.npy
│       └── documents.json
├── models/
│   ├── roberta_bioasq/           # Fine-tuned RoBERTa
│   └── scibert_bioasq/           # Fine-tuned SciBERT
├── results/
│   ├── evaluation_results_*.json # Full evaluation output
│   ├── statistical_analysis.json # Statistical test results
│   ├── hyperparameter_grid_search_*.csv
│   └── figures/                  # Generated visualizations
├── src/
│   ├── __init__.py
│   ├── preprocess_bioasq.py      # Data preprocessing
│   ├── indexer.py                # Index building
│   ├── retriever.py              # Hybrid retrieval
│   ├── qa_models.py              # QA model wrapper
│   ├── train_qa.py               # Model fine-tuning
│   ├── hyperparameter_tuner.py   # Grid search
│   ├── evaluator.py              # Full pipeline evaluation
│   ├── statistical_analysis.py  # Statistical testing
│   ├── data_visualization.py    # Dataset and EDA figurees
│   └── model_visualization.py  # Model Evaluation figures
│   └── analyze_data_quality.py  # Data Quality Analysis
└── README.md
└── requirements.txt
```

## Installation

### Requirements

- Python 3.10+
- macOS with Apple Silicon (M1) or Linux with CUDA
- 16GB RAM minimum

### Setup

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
datasets>=2.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## Usage

### 1. Data Preprocessing

Download BioASQ training data and preprocess:

```bash
# Place training13b.json in data/raw/BioASQ/
python src/preprocess_bioasq.py
```

**Output:** Creates `bioasq_train.json`, `bioasq_val.json`, `bioasq_test.json` in `data/processed/`

### 2. Build Indices

Build BM25 and FAISS indices for hybrid retrieval:

```bash
python src/indexer.py --dataset bioasq
```

**Output:** Creates indices in `data/indices/`

### 3. Fine-tune Models

Fine-tune RoBERTa and SciBERT on BioASQ training data:

```bash
# Fine-tune RoBERTa
python src/train_qa.py \
    --model deepset/roberta-base-squad2 \
    --output models/roberta_bioasq/ \
    --train_file data/processed/bioasq_train.json

# Fine-tune SciBERT
python src/train_qa.py \
    --model ktrapeznikov/scibert_scivocab_uncased_squad_v2 \
    --output models/scibert_bioasq/ \
    --train_file data/processed/bioasq_train.json
```

### 4. Hyperparameter Tuning

Run grid search on validation set:

```bash
# Tune both models
python src/hyperparameter_tuner.py --dataset bioasq --model both
```

**Search space:**
- `top_k`: {5, 10, 15} - passages to retrieve
- `rrf_k`: {30, 60, 90} - RRF fusion parameter
- `qa_top_k`: {3, 5, 10} - passages for QA model

**Output:** `results/best_hyperparameters_roberta.json`, `results/best_hyperparameters_scibert.json`

### 5. Full Evaluation

Run complete evaluation on test set:

```bash
python src/evaluator.py --dataset bioasq
```

**Output:** `results/evaluation_results_YYYYMMDD_HHMMSS.json`, `results/evaluation_data_*.csv`

### 6. Statistical Analysis

Perform comprehensive statistical testing:

```bash
python src/statistical_analysis.py
```

**Tests performed:**
- Paired t-test (F1, Exact Match)
- Bootstrap confidence intervals (1000 iterations)
- Cohen's d effect size
- McNemar's test for classifier agreement
- Latency comparison

**Output:** `results/statistical_analysis.json`

### 7. Generate Visualizations

Create publication-ready figures:

```bash
python src/model_visualizations.py
python src/data_visualizations.py

```

**Output figures:**
- `fig1_f1_comparison.png` - F1 with confidence intervals
- `fig2_f1_distributions.png` - Violin and box plots
- `fig3_scatter_comparison.png` - Per-example comparison
- `fig4_latency_comparison.png` - Latency breakdown
- `fig5_confidence_calibration.png` - Calibration curves
- `fig6_mcnemar_matrix.png` - Agreement matrix
- `fig7_comprehensive_summary.png` - Multi-panel summary

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: HYBRID RETRIEVAL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Question ──┬──► BM25 ────────┬──► Reciprocal ──► Top-K        │
│              │                 │    Rank Fusion    Passages      │
│              └──► FAISS ───────┘    (k=60)                      │
│                   (MiniLM)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 2: EXTRACTIVE QA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Passages ──► RoBERTa / SciBERT ──► Joint Span ──► Answer      │
│                                      Selection                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

Experiments were conducted on:
- **CPU:** Apple M1 Pro (10-core)
- **Memory:** 16GB unified memory
- **Storage:** 512GB SSD

Approximate runtimes:
- Data preprocessing: ~1 minute
- Index building: ~5 minutes
- Model fine-tuning: ~30 minutes per model
- Hyperparameter tuning: ~30 minutes per model
- Full evaluation: ~15 minutes
- Statistical analysis: ~1 minute


## References

Key papers this work builds upon:

- Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. *EMNLP*.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv*.
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR*.
- Tsatsaronis, G., et al. (2015). An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition. *BMC Bioinformatics*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Essex, School of Mathematics, Statistics and Actuarial Science
- Dr. Daniel F. Ahelegbey for supervision
- BioASQ organizers for the dataset
- Hugging Face for pre-trained models and transformers library
