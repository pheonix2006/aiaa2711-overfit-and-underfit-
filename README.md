# Overfitting & Underfitting in Machine Learning

AIAA 2711 — 2025 Spring Group Project

## Overview

This project explores overfitting and underfitting in machine learning through:

- **Bias-Variance Decomposition** — mathematical derivation and Monte Carlo verification
- **Polynomial Regression Experiments** — visual demonstration of underfitting, good fit, and overfitting
- **Regularization Analysis** — Ridge (L2) and Lasso (L1) comparison with coefficient shrinkage visualization

## Project Structure

```
src/                    Core Python modules (all logic lives here)
  data_generation.py      Synthetic data: f(x) = sin(2*pi*x) + noise
  polynomial_regression.py  Polynomial fitting + bias-variance decomposition
  regularization.py       Ridge / Lasso regression wrappers
  visualization.py        All plotting functions
  utils.py                MSE, train/test split

notebooks/              Jupyter notebooks (import from src/, demo only)
  01_polynomial_regression_demo.ipynb
  02_regularization_demo.ipynb

report/                 LaTeX report
  main.tex                Report source
  references.bib          Bibliography (7 references)
  figures/                Generated visualizations (6 PNGs)

tests/                  Unit tests (20 tests)
slides/                 Presentation materials
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Execute notebooks (generates figures into report/figures/)
uv run jupyter nbconvert --execute --inplace notebooks/01_polynomial_regression_demo.ipynb
uv run jupyter nbconvert --execute --inplace notebooks/02_regularization_demo.ipynb

# Compile LaTeX report (requires pdflatex)
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Dependencies

- Python >= 3.11
- numpy, matplotlib, scikit-learn, jupyter, pytest
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
