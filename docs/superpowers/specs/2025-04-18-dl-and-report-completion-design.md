# Deep Learning Module & Report Completion — Design Spec

**Course:** AIAA 2711 — 2025 Spring
**Scope:** Complete remaining deliverables: DL overfitting demo + full report text
**Depends on:** Existing spec `2025-04-18-overfitting-underfitting-design.md` (Tasks 1-9 done)

---

## 1. Goal

Complete two remaining deliverables:
1. **Notebook 03** — Deep learning overfitting demo (PyTorch) with two experiments
2. **Report `main.tex`** — Fill all empty sections with proper academic text, integrating all figures

## 2. New Module: `src/deep_learning.py`

### 2.1 Architecture

Single module following existing patterns: functions return dicts, no global state, deterministic seeds.

**Dependencies:** `torch`, `torchvision` (added to `pyproject.toml`)

### 2.2 Public API

```python
def build_mlp(
    input_dim: int,
    hidden_layers: list[int],
    output_dim: int = 1,
    dropout_rate: float = 0.0,
    activation: str = "relu",
) -> nn.Module:
    """Build configurable MLP. Returns nn.Sequential."""

def train_model(
    model: nn.Module,
    X_train: Tensor, y_train: Tensor,
    X_test: Tensor, y_test: Tensor,
    epochs: int = 500,
    lr: float = 0.01,
    task: str = "regression",  # "regression" or "classification"
) -> dict:
    """Train loop. Returns {train_losses, test_losses, best_epoch, model}."""

def evaluate_network_complexity(
    X_train, y_train, X_test, y_test,
    architectures: list[list[int]],
    seed: int = 42,
    task: str = "regression",
) -> dict:
    """Compare architectures. Returns {architectures, param_counts, train_errors, test_errors, histories}."""

def load_mnist_subset(
    n_train: int = 500,
    n_test: int = 1000,
    seed: int = 42,
) -> tuple:
    """Load MNIST, return (X_train, y_train, X_test, y_test) as tensors.
    Flatten to 784-dim. Normalize to [0,1]."""
```

### 2.3 Design Decisions

- **No custom training loop class** — plain functions, matching existing module style (KISS)
- **`train_model` returns loss history** — enables plotting train/test curves and early stopping analysis
- **Early stopping** — implemented inside `train_model` via patience parameter; returns `best_epoch`
- **`torch.manual_seed(seed)`** for reproducibility
- **Regression uses MSELoss, classification uses CrossEntropyLoss**

## 3. New Visualization Functions in `src/visualization.py`

```python
def plot_dl_synthetic_fits(
    X_train, y_train, predictions_by_arch: dict,
    true_fn=None,
) -> Figure:
    """Plot MLP regression fits for different architectures (shallow vs deep)."""

def plot_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    best_epoch: int | None = None,
    title: str = "Training Curves",
) -> Figure:
    """Plot train/test loss over epochs. Mark early stopping point if provided."""

def plot_dropout_comparison(
    history_no_dropout: dict,
    history_with_dropout: dict,
    title: str = "Dropout Effect",
) -> Figure:
    """Side-by-side: train/test curves with and without dropout."""

def plot_network_complexity_comparison(
    param_counts: list[int],
    train_errors: list[float],
    test_errors: list[float],
) -> Figure:
    """Test error vs. network parameter count."""
```

## 4. Tests: `tests/test_deep_learning.py`

Approximately 5-6 tests:
- `test_build_mlp_output_shape` — verify model produces correct output dimensions
- `test_build_mlp_with_dropout` — verify dropout layers are present
- `test_train_model_returns_history` — verify dict structure
- `test_train_model_loss_decreases` — train loss should decrease over epochs
- `test_evaluate_network_complexity_structure` — verify return dict
- `test_load_mnist_subset_shapes` — verify dimensions (500, 784) etc.

## 5. Notebook 03: Deep Learning Overfitting Demo

### Structure

**Part A: Regression on sin(2πx)**
1. Generate same synthetic data as notebook 01
2. Fit MLP with different architectures:
   - Shallow: `[8]` (8 hidden units, 1 layer)
   - Medium: `[32, 16]` (good fit)
   - Deep: `[128, 64, 32, 16]` (overfitting)
3. Plot fitting curves → `dl_synthetic_overfitting.png`
4. Plot train/test loss curves with early stopping → `dl_training_curves.png`

**Part B: Classification on MNIST subset**
1. Load 500 training + 1000 test samples
2. Train MLP `[256, 128]` without dropout — observe overfitting
3. Train same architecture with `dropout=0.5` — observe regularization effect
4. Plot comparison → `dl_dropout_effect.png`

**Part C: Network Complexity vs. Generalization**
1. Evaluate architectures from tiny to large on synthetic data
2. Plot param count vs. test error → `dl_network_complexity.png`

### Figures Generated (4 new)

| Figure File | Content |
|-------------|---------|
| `dl_synthetic_overfitting.png` | MLP fits on sin(2πx): shallow/medium/deep |
| `dl_training_curves.png` | Loss curves + early stopping marker |
| `dl_dropout_effect.png` | MNIST train/test with vs. without Dropout |
| `dl_network_complexity.png` | Parameter count vs. test error |

## 6. Report `main.tex` — Full Text

All sections currently contain only LaTeX comments. Fill with proper academic English text.

### Section Plan

**Section 1: Introduction** (~300 words)
- Machine learning success depends on generalization
- Define overfitting and underfitting informally
- Preview report structure
- Cite \cite{bishop2006pattern}, \cite{ying2019overview}

**Section 2: Theoretical Foundation** (~600 words)
- 2.1 Formal definitions: training error vs. test error, generalization gap
- 2.2 Full bias-variance decomposition derivation (step by step from expected MSE)
- 2.3 Model complexity analysis: how bias/variance shift as complexity grows
- Cite \cite{hastie2009elements}, \cite{bishop2006pattern}

**Section 3: Experimental Design and Results** (~500 words)
- 3.1 Setup: describe data generation (sin(2πx) + noise), polynomial model family
- 3.2 Analysis of Figure 1 (polynomial_fits): describe what each degree shows
- 3.3 Analysis of Figure 2 (error_vs_complexity): U-shaped test error curve
- 3.4 Analysis of Figure 3 (bias_variance_tradeoff): empirical verification of theory

**Section 4: Solutions: Regularization** (~500 words)
- 4.1 Ridge: closed-form solution derivation, geometric interpretation, analysis of Figures 4-5
- 4.2 Lasso: sparsity property explanation, comparison with Ridge via Figure 6
- 4.3 Cross-validation: K-fold principle, how it selects optimal α
- 4.4 Other techniques: Early Stopping (with training curve intuition), Dropout (Bernoulli mask)
- Cite \cite{hoerl1970ridge}, \cite{tibshirani1996regression}, \cite{srivastava2014dropout}

**Section 5: Application in AI — Deep Learning** (~500 words)
- Overfitting in deep neural networks: over-parameterization paradox
- Network depth/width vs. overfitting (reference Figure 7: dl_synthetic_overfitting)
- Training dynamics: early stopping in DL (reference Figure 8: dl_training_curves)
- Dropout as regularization (reference Figure 9: dl_dropout_effect)
- Network complexity vs. generalization (reference Figure 10: dl_network_complexity)
- Brief mention: data augmentation, weight decay, batch normalization
- Cite \cite{goodfellow2016deep}, \cite{srivastava2014dropout}

**Section 6: Conclusion** (~200 words)
- Summary of key findings
- Practical guidelines: start simple, use validation, apply regularization
- Connection between classical theory and modern deep learning

### Figure Numbering (Final)

| # | File | Caption |
|---|------|---------|
| 1 | polynomial_fits.png | Polynomial fits at degree 1, 4, 15 |
| 2 | error_vs_complexity.png | Training and test error vs. polynomial degree |
| 3 | bias_variance_tradeoff.png | Empirical bias-variance tradeoff |
| 4 | ridge_regularization.png | Ridge: error and coefficient norm vs. α |
| 5 | lasso_regularization.png | Lasso: error and coefficient norm vs. α |
| 6 | ridge_vs_lasso_coefficients.png | Ridge vs. Lasso coefficient comparison |
| 7 | dl_synthetic_overfitting.png | MLP regression: shallow vs. deep network fits |
| 8 | dl_training_curves.png | DL training curves with early stopping |
| 9 | dl_dropout_effect.png | Dropout effect on MNIST classification |
| 10 | dl_network_complexity.png | Network parameter count vs. test error |

### References to Add

```bibtex
@article{zhang2021understanding,
  title={Understanding Deep Learning (Still) Requires Rethinking Generalization},
  author={Zhang, Chiyuan and Bengio, Samy and Hardt, Moritz and Recht, Benjamin and Vinyals, Oriol},
  journal={Communications of the ACM},
  volume={64},
  number={3},
  pages={107--115},
  year={2021}
}
```

## 7. Updated File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add torch, torchvision |
| `src/deep_learning.py` | Create | DL model building, training, evaluation |
| `src/visualization.py` | Modify | Add 4 DL plotting functions |
| `tests/test_deep_learning.py` | Create | 5-6 tests for DL module |
| `notebooks/03_deep_learning_overfitting.ipynb` | Create | Full DL demo notebook |
| `report/main.tex` | Modify | Fill all empty sections with text |
| `report/references.bib` | Modify | Add Zhang et al. 2021 |
| `report/figures/dl_*.png` | Create | 4 new figures from notebook 03 |

## 8. Constraints

- All DL code in `src/deep_learning.py`, notebook only imports and calls
- Use `torch.manual_seed()` for reproducibility
- Keep MLP simple (no CNN, no complex architectures) — focus is on overfitting concept
- MNIST subset (500 train) intentionally small to trigger overfitting
- Report text in English, academic tone
- All figures must be referenced and discussed in the report text (rubric: "图文结合")
