# Overfitting & Underfitting in Machine Learning

AIAA 2711 — 2025 Spring Group Project

## Overview

This project explores overfitting and underfitting in machine learning through:

- **Bias-Variance Decomposition** — mathematical derivation and Monte Carlo verification
- **Polynomial Regression Experiments** — visual demonstration of underfitting, good fit, and overfitting
- **Regularization Analysis** — Ridge (L2) and Lasso (L1) comparison with coefficient shrinkage visualization
- **Deep Learning Experiments** — MLP overfitting on synthetic data and MNIST, with Dropout and Early Stopping

## Project Structure

```
src/                    Core Python modules (all logic lives here)
  data_generation.py      Synthetic data: f(x) = sin(2*pi*x) + noise
  polynomial_regression.py  Polynomial fitting + bias-variance decomposition
  regularization.py       Ridge / Lasso regression wrappers
  deep_learning.py        MLP building, training, MNIST loading (PyTorch)
  visualization.py        All plotting functions
  utils.py                MSE, train/test split

notebooks/              Jupyter notebooks (import from src/, demo only)
  01_polynomial_regression_demo.ipynb
  02_regularization_demo.ipynb
  03_deep_learning_overfitting.ipynb

report/                 LaTeX report
  main.tex                Report source
  references.bib          Bibliography (8 references)
  figures/                Generated visualizations (10 PNGs)

tests/                  Unit tests (27 tests)
slides/                 Presentation materials
data/                   Downloaded datasets (gitignored)
```

## Quick Start

```bash
# Install dependencies (includes PyTorch)
uv sync

# Run all tests
uv run pytest tests/ -v

# Execute notebooks (generates figures into report/figures/)
uv run jupyter nbconvert --execute --inplace notebooks/01_polynomial_regression_demo.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/02_regularization_demo.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/03_deep_learning_overfitting.ipynb

# Compile LaTeX report (requires pdflatex)
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## MNIST Data

Notebook 03 uses the MNIST dataset (~12 MB), which is **auto-downloaded** on first run and cached in `data/` (gitignored). If auto-download fails, see the manual download instructions in the notebook cell comments.

## Dependencies

- Python >= 3.11
- numpy, matplotlib, scikit-learn, jupyter, pytest
- torch, torchvision (for deep learning experiments)
- Managed via [uv](https://docs.astral.sh/uv/)

## Generated Figures

| Figure | Description |
|--------|-------------|
| `polynomial_fits.png` | Degree 1 / 4 / 15 fitting comparison |
| `error_vs_complexity.png` | Train vs. test error curve |
| `bias_variance_tradeoff.png` | Bias^2 + Variance + noise decomposition |
| `ridge_regularization.png` | Ridge: error & coefficient norm vs. alpha |
| `lasso_regularization.png` | Lasso: error & coefficient norm vs. alpha |
| `ridge_vs_lasso_coefficients.png` | Ridge vs. Lasso coefficient bar chart |
| `dl_synthetic_overfitting.png` | MLP regression: shallow vs. deep fits |
| `dl_training_curves.png` | Training curves with early stopping |
| `dl_dropout_effect.png` | MNIST: with vs. without Dropout |
| `dl_network_complexity.png` | Parameter count vs. test error |
