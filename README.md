# A Comparative Evaluation of Feature Scaling Techniques Across Multiple Classifiers and Domains

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](pyproject.toml)

## Overview

This repository contains the full code, data loaders, experimental results, and manuscript for a comparative study of six feature scaling techniques across seven classifiers and six benchmark datasets (both binary and multi-class), validated with 5-fold stratified cross-validation.

**Key findings:**
- Z-Score Standardization ranks first overall (0.8561), statistically tied with Min-Max (0.8530, p=0.123)
- All scalers significantly outperform raw data (p<0.01)
- Tree-based models are nearly scaling-invariant (<0.2% improvement); SVM (RBF) improves by +7.7%
- Multi-class results confirm findings generalize beyond binary classification

**Scalers evaluated:** Raw, Min-Max, Z-Score, Robust (IQR), MaxAbs, Quantile  
**Classifiers:** k-NN (Euclidean, Manhattan, Cosine), SVM (Linear, RBF), Decision Tree, Random Forest  
**Datasets:** Pima Diabetes, Breast Cancer Wisconsin, Wine, Digits, Yeast, Segment

## Results

| Scaler | Mean Accuracy |
|--------|:------------:|
| **Z-Score** | **0.8561** |
| Min-Max | 0.8530 |
| MaxAbs | 0.8506 |
| Robust | 0.8370 |
| Quantile | 0.8345 |
| Raw | 0.8050 |

Full paper: [results/paper.pdf](results/paper.pdf) (14 pages)  
LaTeX source: [results/paper.tex](results/paper.tex)

## Reproduction

```bash
# Install dependencies
uv sync

# Run full experiment (3,384 configs, 5-fold CV)
uv run python3 -m src.feature_scaling_study.experiments.runner

# Run statistical tests
uv run python3 -m src.feature_scaling_study.experiments.stats

# Generate figures
uv run python3 -m src.feature_scaling_study.visualization.plots

# Compile the paper (requires tectonic)
cd results && tectonic paper.tex
```

## Interactive Dashboard

```bash
uv run marimo run dashboard.py
```

## Citation

```bibtex
@article{wahyudin2026scaling,
  author  = {A. R. Wahyudin and S. Murdiawati and J. A. Putra and E. R. Susanto},
  title   = {A Comparative Evaluation of Feature Scaling Techniques Across Multiple Classifiers and Domains},
  year    = {2026},
  note    = {Unpublished manuscript}
}
```

## License

MIT
