# Overfitting & Underfitting in Machine Learning — Project Design Spec

**Course:** AIAA 2711 — 2025 Spring
**Topic:** Overfitting and Underfitting in Machine Learning
**Team Size:** 3
**Deliverables:** Presentation (10min + 3min Q&A), Written Report (LaTeX), Code + Notebooks

---

## 1. Approach

Theory-driven + experimental verification. Mathematical foundations (Bias-Variance Decomposition) as the main thread, with code experiments and visualizations serving as supporting evidence.

## 2. Content Structure

### 2.1 Concept Definitions
- Overfitting: high training performance, poor generalization
- Underfitting: insufficient model complexity, fails to capture patterns
- Training error vs. test error

### 2.2 Mathematical Foundation
- Bias-Variance Decomposition: `E[(y - f̂(x))²] = Bias² + Variance + σ²`
- Underfitting = high bias, Overfitting = high variance
- Model Complexity vs. Bias/Variance tradeoff curve

### 2.3 Experimental Demonstration
- Polynomial regression demo: degree=1 (underfit), degree=4 (good fit), degree=15 (overfit)
- Visualizations: fitting curves comparison, train/test error vs. complexity curve

### 2.4 Solutions
- Regularization: L1 (Lasso) / L2 (Ridge) — mathematical formulation and intuition
- Cross-validation principle
- Early Stopping, Dropout (brief)

### 2.5 Application in AI
- Overfitting in deep learning (relationship between network depth and overfitting)
- Real-world scenario examples

## 3. Presentation Plan (10 min + 3 min Q&A)

| Person | Duration | Content |
|--------|----------|---------|
| A | 4 min | Intuitive intro → Definitions → Bias-Variance Decomposition derivation → Tradeoff curve |
| B | 3 min | Polynomial regression demo → Fitting curve comparison → Train/Test error curve → Link back to math |
| C | 3 min | Regularization math (L1/L2) → Cross-validation → DL overfitting & solutions → Summary |
| All | 3 min | Q&A |

## 4. Report Structure (LaTeX)

1. **Introduction** — Background, problem definition, report objective
2. **Theoretical Foundation** — Full Bias-Variance Decomposition derivation, Model Complexity analysis
3. **Experimental Design & Results** — Experiment setup, code explanation, visualizations and analysis
4. **Solutions & Regularization** — L1/L2 derivation, Cross-validation, Early Stopping/Dropout
5. **Application in AI** — Deep learning scenarios, practical cases
6. **Conclusion**
7. **References**

## 5. Project Structure

```
E:/Project/2711/
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── data_generation.py        # Synthetic data generation
│   ├── polynomial_regression.py  # Polynomial regression fitting & evaluation
│   ├── regularization.py         # L1/L2 regularization implementation
│   ├── visualization.py          # All plotting functions
│   └── utils.py                  # Shared utilities (metrics, etc.)
├── notebooks/
│   ├── 01_polynomial_regression_demo.ipynb   # Overfitting/underfitting demo
│   ├── 02_regularization_demo.ipynb          # Regularization effect demo
│   └── 03_deep_learning_overfitting.ipynb    # (Optional) DL overfitting demo
├── report/
│   ├── main.tex                  # LaTeX main file
│   ├── references.bib            # Bibliography
│   └── figures/                  # Generated figures for report
├── slides/                       # Presentation materials
├── docs/superpowers/specs/       # This spec
├── pyproject.toml
└── uv.lock
```

### Key Principle: Notebooks import from `src/`

Notebooks are for **demonstration and visualization only**. All concrete implementations (data generation, model fitting, regularization, plotting functions) live in `src/`. Notebooks call `from src.xxx import ...`.

## 6. Dependencies

- **Core:** numpy, matplotlib, scikit-learn
- **Optional (DL demo):** pytorch
- **Report:** LaTeX distribution (local or Overleaf)

## 7. Mapping to Rubrics

### Teacher Team Evaluation (100 pts)
- Mathematical Rigor (40): Covered by Section 2.2 (Bias-Variance derivation) + 2.4 (Regularization math)
- Clarity of Expression (30): Diagrams from visualization module + structured presentation flow
- Overall Coherence (30): Theory → Experiment → Solution logical chain + innovative analogies

### Report Evaluation (100 pts)
- Content Quality (45): Individual contribution (per-person sections) + Concept intro + AI application
- Expression and Format (30): LaTeX formatting + clear language
- Overall Evaluation (25): Proper references + professional presentation

### Peer Evaluation (50 pts)
- Clarity of Expression (25): Rehearsed verbal delivery + well-designed slides + time management
- Concept Explanation & Interaction (25): Clear math explanation + Q&A preparation
