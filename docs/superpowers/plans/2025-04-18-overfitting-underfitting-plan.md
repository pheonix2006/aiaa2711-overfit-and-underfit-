# Overfitting & Underfitting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete course project (code + notebooks + LaTeX report + slides structure) demonstrating overfitting/underfitting in ML through theory-driven experimentation.

**Architecture:** Core logic lives in `src/` as importable Python modules. Jupyter notebooks in `notebooks/` import from `src/` for demonstration only. LaTeX report in `report/`. All visualizations generated programmatically via `src/visualization.py` and saved to `report/figures/`.

**Tech Stack:** Python 3.11+, numpy, matplotlib, scikit-learn, uv (dependency management), LaTeX

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/__init__.py` | Package init, expose public API |
| `src/data_generation.py` | Generate synthetic datasets (true function + noise) |
| `src/polynomial_regression.py` | Polynomial fitting, train/test evaluation, bias-variance computation |
| `src/regularization.py` | Ridge (L2) and Lasso (L1) regression wrappers with coefficient extraction |
| `src/visualization.py` | All plotting: fit curves, error curves, bias-variance tradeoff, regularization effects |
| `src/utils.py` | Metrics (MSE, etc.), train/test split helper |
| `tests/test_data_generation.py` | Tests for data generation |
| `tests/test_polynomial_regression.py` | Tests for polynomial fitting and evaluation |
| `tests/test_regularization.py` | Tests for regularization |
| `tests/test_utils.py` | Tests for utility functions |
| `notebooks/01_polynomial_regression_demo.ipynb` | Overfitting/underfitting visual demo |
| `notebooks/02_regularization_demo.ipynb` | Regularization effect demo |
| `notebooks/03_deep_learning_overfitting.ipynb` | (Optional) DL overfitting demo |
| `report/main.tex` | LaTeX report main file |
| `report/references.bib` | Bibliography |
| `report/figures/` | Generated figure output directory |

---

## Task 1: Project Dependencies & Package Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add dependencies via uv**

```bash
cd E:/Project/2711
uv add numpy matplotlib scikit-learn jupyter pytest
```

- [ ] **Step 2: Create src package**

Create `src/__init__.py`:

```python
"""Overfitting & Underfitting in ML — Core modules."""
```

- [ ] **Step 3: Create tests package**

Create `tests/__init__.py`:

```python
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p notebooks report/figures slides
```

- [ ] **Step 5: Verify setup**

```bash
uv run python -c "import numpy; import matplotlib; import sklearn; print('All dependencies OK')"
```

Expected: `All dependencies OK`

- [ ] **Step 6: Commit**

```bash
git add src/ tests/ pyproject.toml uv.lock notebooks/.gitkeep report/.gitkeep slides/.gitkeep
git commit -m "feat: add core dependencies and project structure"
```

---

## Task 2: Utility Module

**Files:**
- Create: `src/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_utils.py`:

```python
import numpy as np
from src.utils import mse, train_test_split_data


def test_mse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    # MSE = (0 + 0 + 4) / 3
    assert abs(mse(y_true, y_pred) - 4.0 / 3.0) < 1e-10


def test_train_test_split_sizes():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_ratio=0.2, seed=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_train_test_split_reproducible():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    split1 = train_test_split_data(X, y, test_ratio=0.2, seed=42)
    split2 = train_test_split_data(X, y, test_ratio=0.2, seed=42)
    np.testing.assert_array_equal(split1[0], split2[0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils'`

- [ ] **Step 3: Implement src/utils.py**

Create `src/utils.py`:

```python
"""Shared utility functions: metrics and data splitting."""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def train_test_split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        test_ratio: Fraction of data used for testing.
        seed: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add utility module with MSE and train/test split"
```

---

## Task 3: Data Generation Module

**Files:**
- Create: `src/data_generation.py`
- Create: `tests/test_data_generation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_generation.py`:

```python
import numpy as np
from src.data_generation import generate_polynomial_data, true_function


def test_true_function_shape():
    x = np.linspace(0, 1, 50)
    y = true_function(x)
    assert y.shape == (50,)


def test_generate_polynomial_data_shape():
    X, y = generate_polynomial_data(n_samples=100, noise_std=0.1, seed=42)
    assert X.shape == (100,)
    assert y.shape == (100,)


def test_generate_polynomial_data_reproducible():
    X1, y1 = generate_polynomial_data(n_samples=50, noise_std=0.1, seed=42)
    X2, y2 = generate_polynomial_data(n_samples=50, noise_std=0.1, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_polynomial_data_no_noise():
    X, y = generate_polynomial_data(n_samples=50, noise_std=0.0, seed=42)
    expected = true_function(X)
    np.testing.assert_array_almost_equal(y, expected)


def test_generate_polynomial_data_has_noise():
    X, y = generate_polynomial_data(n_samples=100, noise_std=1.0, seed=42)
    expected = true_function(X)
    # With noise_std=1.0, values should differ noticeably
    assert np.mean(np.abs(y - expected)) > 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_data_generation.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement src/data_generation.py**

Create `src/data_generation.py`:

```python
"""Synthetic data generation for overfitting/underfitting demos."""

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """The ground-truth function: f(x) = sin(2*pi*x).

    This is the classic function used in Bishop's PRML for
    demonstrating polynomial curve fitting and overfitting.
    """
    return np.sin(2 * np.pi * x)


def generate_polynomial_data(
    n_samples: int = 30,
    noise_std: float = 0.3,
    seed: int = 42,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy samples from the true function.

    Args:
        n_samples: Number of data points.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.
        x_range: (min, max) range for x values.

    Returns:
        (X, y) where X is uniformly sampled and y = f(X) + noise.
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(x_range[0], x_range[1], n_samples)
    noise = rng.normal(0, noise_std, size=n_samples) if noise_std > 0 else np.zeros(n_samples)
    y = true_function(X) + noise
    return X, y
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_data_generation.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation.py tests/test_data_generation.py
git commit -m "feat: add data generation module with sin(2*pi*x) ground truth"
```

---

## Task 4: Polynomial Regression Module

**Files:**
- Create: `src/polynomial_regression.py`
- Create: `tests/test_polynomial_regression.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_polynomial_regression.py`:

```python
import numpy as np
from src.polynomial_regression import (
    fit_polynomial,
    predict_polynomial,
    evaluate_polynomial_degrees,
    compute_bias_variance,
)


def test_fit_polynomial_linear():
    # y = 2x + 1, degree=1 should fit exactly
    X = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * X + 1.0
    coeffs = fit_polynomial(X, y, degree=1)
    assert len(coeffs) == 2  # degree+1 coefficients


def test_predict_polynomial():
    X = np.array([0.0, 1.0, 2.0])
    y = 2.0 * X + 1.0
    coeffs = fit_polynomial(X, y, degree=1)
    y_pred = predict_polynomial(X, coeffs)
    np.testing.assert_array_almost_equal(y_pred, y, decimal=10)


def test_fit_polynomial_high_degree():
    X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.sin(2 * np.pi * X)
    coeffs = fit_polynomial(X, y, degree=4)
    # With 5 points and degree 4, should fit perfectly
    y_pred = predict_polynomial(X, coeffs)
    np.testing.assert_array_almost_equal(y_pred, y, decimal=5)


def test_evaluate_polynomial_degrees_returns_dict():
    X_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * X_train)
    X_test = np.linspace(0, 1, 10)
    y_test = np.sin(2 * np.pi * X_test)
    degrees = [1, 3, 5]
    results = evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, degrees)
    assert set(results.keys()) == {"degrees", "train_errors", "test_errors", "coefficients"}
    assert len(results["train_errors"]) == 3
    assert len(results["test_errors"]) == 3


def test_evaluate_overfitting_pattern():
    """High degree should have low train error but high test error."""
    rng = np.random.default_rng(42)
    X_train = np.linspace(0, 1, 15)
    y_train = np.sin(2 * np.pi * X_train) + rng.normal(0, 0.3, 15)
    X_test = np.linspace(0, 1, 50)
    y_test = np.sin(2 * np.pi * X_test)
    results = evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, [1, 4, 14])
    # degree=14 train error should be lowest
    assert results["train_errors"][2] < results["train_errors"][0]
    # degree=14 test error should be higher than degree=4
    assert results["test_errors"][2] > results["test_errors"][1]


def test_compute_bias_variance_shapes():
    result = compute_bias_variance(
        true_fn=lambda x: np.sin(2 * np.pi * x),
        n_samples=20,
        noise_std=0.3,
        degrees=[1, 3, 5, 9],
        n_repeats=50,
        seed=42,
    )
    assert len(result["degrees"]) == 4
    assert len(result["bias_squared"]) == 4
    assert len(result["variance"]) == 4
    assert len(result["total_error"]) == 4


def test_compute_bias_variance_tradeoff():
    """Low degree = high bias; high degree = high variance."""
    result = compute_bias_variance(
        true_fn=lambda x: np.sin(2 * np.pi * x),
        n_samples=20,
        noise_std=0.3,
        degrees=[1, 9],
        n_repeats=100,
        seed=42,
    )
    # degree=1 should have higher bias than degree=9
    assert result["bias_squared"][0] > result["bias_squared"][1]
    # degree=9 should have higher variance than degree=1
    assert result["variance"][1] > result["variance"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_polynomial_regression.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement src/polynomial_regression.py**

Create `src/polynomial_regression.py`:

```python
"""Polynomial regression: fitting, prediction, and bias-variance decomposition."""

from typing import Callable

import numpy as np

from src.utils import mse


def fit_polynomial(X: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit a polynomial of given degree using least squares.

    Args:
        X: Input values (n_samples,).
        y: Target values (n_samples,).
        degree: Polynomial degree.

    Returns:
        Coefficient array from np.polyfit (highest degree first).
    """
    return np.polyfit(X, y, degree)


def predict_polynomial(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Predict using polynomial coefficients.

    Args:
        X: Input values (n_samples,).
        coeffs: Coefficients from fit_polynomial.

    Returns:
        Predicted values (n_samples,).
    """
    return np.polyval(coeffs, X)


def evaluate_polynomial_degrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degrees: list[int],
) -> dict:
    """Fit polynomials of various degrees and compute train/test errors.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        degrees: List of polynomial degrees to evaluate.

    Returns:
        Dict with keys: degrees, train_errors, test_errors, coefficients.
    """
    train_errors = []
    test_errors = []
    all_coeffs = []

    for d in degrees:
        coeffs = fit_polynomial(X_train, y_train, d)
        y_train_pred = predict_polynomial(X_train, coeffs)
        y_test_pred = predict_polynomial(X_test, coeffs)
        train_errors.append(mse(y_train, y_train_pred))
        test_errors.append(mse(y_test, y_test_pred))
        all_coeffs.append(coeffs)

    return {
        "degrees": degrees,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "coefficients": all_coeffs,
    }


def compute_bias_variance(
    true_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 20,
    noise_std: float = 0.3,
    degrees: list[int] | None = None,
    n_repeats: int = 200,
    seed: int = 42,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> dict:
    """Estimate bias^2, variance, and total error via Monte Carlo simulation.

    For each degree, generates n_repeats datasets, fits a polynomial each time,
    and computes bias^2 and variance of predictions at fixed test points.

    Args:
        true_fn: Ground-truth function f(x).
        n_samples: Samples per generated dataset.
        noise_std: Noise standard deviation.
        degrees: List of polynomial degrees to evaluate.
        n_repeats: Number of Monte Carlo repetitions.
        seed: Random seed.
        x_range: (min, max) for data generation.

    Returns:
        Dict with keys: degrees, bias_squared, variance, total_error.
    """
    if degrees is None:
        degrees = [1, 2, 3, 5, 7, 9, 12]

    rng = np.random.default_rng(seed)
    x_test = np.linspace(x_range[0], x_range[1], 100)
    y_true = true_fn(x_test)

    bias_squared_list = []
    variance_list = []
    total_error_list = []

    for d in degrees:
        predictions = np.zeros((n_repeats, len(x_test)))

        for i in range(n_repeats):
            x_train = np.sort(rng.uniform(x_range[0], x_range[1], n_samples))
            y_train = true_fn(x_train) + rng.normal(0, noise_std, n_samples)
            coeffs = fit_polynomial(x_train, y_train, d)
            predictions[i] = predict_polynomial(x_test, coeffs)

        mean_pred = predictions.mean(axis=0)
        bias_sq = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(predictions.var(axis=0))
        total = bias_sq + variance + noise_std ** 2

        bias_squared_list.append(bias_sq)
        variance_list.append(variance)
        total_error_list.append(total)

    return {
        "degrees": degrees,
        "bias_squared": bias_squared_list,
        "variance": variance_list,
        "total_error": total_error_list,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_polynomial_regression.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/polynomial_regression.py tests/test_polynomial_regression.py
git commit -m "feat: add polynomial regression module with bias-variance decomposition"
```

---

## Task 5: Regularization Module

**Files:**
- Create: `src/regularization.py`
- Create: `tests/test_regularization.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_regularization.py`:

```python
import numpy as np
from src.regularization import (
    fit_ridge,
    fit_lasso,
    evaluate_regularization,
)


def test_fit_ridge_returns_predictions():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel())
    result = fit_ridge(X, y, alpha=1.0)
    assert "predictions" in result
    assert "coefficients" in result
    assert "model" in result
    assert len(result["predictions"]) == len(y)


def test_fit_lasso_returns_predictions():
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel())
    result = fit_lasso(X, y, alpha=0.1)
    assert "predictions" in result
    assert "coefficients" in result
    assert len(result["predictions"]) == len(y)


def test_ridge_stronger_alpha_shrinks_coefficients():
    """Higher alpha should shrink coefficients more."""
    from sklearn.preprocessing import PolynomialFeatures
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel()) + np.random.default_rng(42).normal(0, 0.3, 30)
    X_poly = PolynomialFeatures(degree=10, include_bias=False).fit_transform(X)

    result_low = fit_ridge(X_poly, y, alpha=0.001)
    result_high = fit_ridge(X_poly, y, alpha=100.0)
    assert np.linalg.norm(result_high["coefficients"]) < np.linalg.norm(result_low["coefficients"])


def test_evaluate_regularization_structure():
    X_train = np.linspace(0, 1, 30).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * X_train.ravel())
    X_test = np.linspace(0, 1, 10).reshape(-1, 1)
    y_test = np.sin(2 * np.pi * X_test.ravel())
    alphas = [0.001, 0.1, 1.0]

    result = evaluate_regularization(
        X_train, y_train, X_test, y_test,
        degree=10, alphas=alphas, method="ridge",
    )
    assert len(result["alphas"]) == 3
    assert len(result["train_errors"]) == 3
    assert len(result["test_errors"]) == 3
    assert len(result["coefficient_norms"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_regularization.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement src/regularization.py**

Create `src/regularization.py`:

```python
"""Regularization: Ridge (L2) and Lasso (L1) regression."""

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

from src.utils import mse


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Fit Ridge (L2) regression.

    Args:
        X: Feature matrix (n_samples, n_features). Should be pre-transformed
           (e.g., polynomial features) if needed.
        y: Target values (n_samples,).
        alpha: Regularization strength.

    Returns:
        Dict with keys: predictions, coefficients, model.
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return {
        "predictions": model.predict(X),
        "coefficients": model.coef_,
        "model": model,
    }


def fit_lasso(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> dict:
    """Fit Lasso (L1) regression.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target values (n_samples,).
        alpha: Regularization strength.

    Returns:
        Dict with keys: predictions, coefficients, model.
    """
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X, y)
    return {
        "predictions": model.predict(X),
        "coefficients": model.coef_,
        "model": model,
    }


def evaluate_regularization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    degree: int = 10,
    alphas: list[float] | None = None,
    method: str = "ridge",
) -> dict:
    """Evaluate regularization across different alpha values.

    Applies polynomial feature expansion, then fits Ridge or Lasso
    for each alpha, recording train/test errors and coefficient norms.

    Args:
        X_train, y_train: Training data (X should be (n, 1)).
        X_test, y_test: Test data.
        degree: Polynomial degree for feature expansion.
        alphas: List of regularization strengths to try.
        method: "ridge" or "lasso".

    Returns:
        Dict with keys: alphas, train_errors, test_errors, coefficient_norms.
    """
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    fit_fn = fit_ridge if method == "ridge" else fit_lasso
    train_errors = []
    test_errors = []
    coefficient_norms = []

    for alpha in alphas:
        result = fit_fn(X_train_poly, y_train, alpha=alpha)
        model = result["model"]
        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)
        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test, test_pred))
        coefficient_norms.append(float(np.linalg.norm(result["coefficients"])))

    return {
        "alphas": alphas,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "coefficient_norms": coefficient_norms,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_regularization.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/regularization.py tests/test_regularization.py
git commit -m "feat: add regularization module with Ridge and Lasso"
```

---

## Task 6: Visualization Module

**Files:**
- Create: `src/visualization.py`

- [ ] **Step 1: Implement src/visualization.py**

Create `src/visualization.py`:

```python
"""Visualization functions for overfitting/underfitting demonstrations.

All functions accept data/results and return matplotlib Figure objects.
Figures can be displayed in notebooks or saved to report/figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# -- Style configuration --
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 10,
}


def apply_style():
    """Apply consistent plot style."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_polynomial_fits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coefficients_by_degree: dict[int, np.ndarray],
    true_fn=None,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> Figure:
    """Plot data points and polynomial fits for multiple degrees.

    Args:
        X_train: Training x values.
        y_train: Training y values.
        coefficients_by_degree: {degree: coefficients} dict.
        true_fn: Optional ground-truth function to overlay.
        x_range: Range for plotting the fit curves.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    n_plots = len(coefficients_by_degree)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    x_smooth = np.linspace(x_range[0], x_range[1], 200)

    for idx, (degree, coeffs) in enumerate(coefficients_by_degree.items()):
        ax = axes[0, idx]
        ax.scatter(X_train, y_train, c="steelblue", s=30, alpha=0.7, label="Training data", zorder=3)

        if true_fn is not None:
            ax.plot(x_smooth, true_fn(x_smooth), "g--", alpha=0.6, label="True function")

        y_fit = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_fit, "r-", label=f"Degree {degree}")
        ax.set_title(f"Polynomial Degree = {degree}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-2, 2)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_error_vs_complexity(
    degrees: list[int],
    train_errors: list[float],
    test_errors: list[float],
) -> Figure:
    """Plot train and test error as a function of model complexity (degree).

    Args:
        degrees: Polynomial degrees.
        train_errors: Training MSE for each degree.
        test_errors: Test MSE for each degree.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, train_errors, "o-", color="steelblue", label="Training Error")
    ax.plot(degrees, test_errors, "s-", color="tomato", label="Test Error")
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Training vs. Test Error")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bias_variance_tradeoff(
    degrees: list[int],
    bias_squared: list[float],
    variance: list[float],
    total_error: list[float],
) -> Figure:
    """Plot bias^2, variance, and total error vs. model complexity.

    Args:
        degrees: Polynomial degrees.
        bias_squared: Bias^2 for each degree.
        variance: Variance for each degree.
        total_error: Total expected error for each degree.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, bias_squared, "o-", color="steelblue", label="Bias²")
    ax.plot(degrees, variance, "s-", color="tomato", label="Variance")
    ax.plot(degrees, total_error, "^-", color="seagreen", label="Total Error (Bias² + Var + σ²)")
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("Error")
    ax.set_title("Bias-Variance Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate underfitting / overfitting zones
    mid = len(degrees) // 2
    ax.axvspan(degrees[0], degrees[mid], alpha=0.05, color="blue", label="_Underfitting zone")
    ax.axvspan(degrees[mid], degrees[-1], alpha=0.05, color="red", label="_Overfitting zone")
    ax.text(degrees[1], max(total_error) * 0.9, "Underfitting\n(High Bias)", fontsize=9, color="blue")
    ax.text(degrees[-2], max(total_error) * 0.9, "Overfitting\n(High Variance)", fontsize=9, color="red", ha="right")

    fig.tight_layout()
    return fig


def plot_regularization_effect(
    alphas: list[float],
    train_errors: list[float],
    test_errors: list[float],
    coefficient_norms: list[float],
    method: str = "Ridge",
) -> Figure:
    """Plot regularization effect: error and coefficient norms vs. alpha.

    Args:
        alphas: Regularization strength values.
        train_errors: Training MSE for each alpha.
        test_errors: Test MSE for each alpha.
        coefficient_norms: L2 norm of coefficients for each alpha.
        method: "Ridge" or "Lasso" (for labeling).

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Error vs alpha
    ax1.plot(alphas, train_errors, "o-", color="steelblue", label="Training Error")
    ax1.plot(alphas, test_errors, "s-", color="tomato", label="Test Error")
    ax1.set_xscale("log")
    ax1.set_xlabel(f"{method} α (Regularization Strength)")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title(f"{method}: Error vs. Regularization Strength")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Coefficient norm vs alpha
    ax2.plot(alphas, coefficient_norms, "D-", color="mediumpurple")
    ax2.set_xscale("log")
    ax2.set_xlabel(f"{method} α (Regularization Strength)")
    ax2.set_ylabel("‖w‖₂ (Coefficient Norm)")
    ax2.set_title(f"{method}: Coefficient Shrinkage")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_ridge_vs_lasso_coefficients(
    ridge_coefficients: np.ndarray,
    lasso_coefficients: np.ndarray,
    feature_names: list[str] | None = None,
) -> Figure:
    """Side-by-side comparison of Ridge and Lasso coefficient values.

    Args:
        ridge_coefficients: Coefficient array from Ridge.
        lasso_coefficients: Coefficient array from Lasso.
        feature_names: Optional feature labels.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    n = len(ridge_coefficients)
    if feature_names is None:
        feature_names = [f"x^{i+1}" for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x_pos = np.arange(n)

    ax1.bar(x_pos, ridge_coefficients, color="steelblue", alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, fontsize=8)
    ax1.set_title("Ridge (L2) Coefficients")
    ax1.set_ylabel("Coefficient Value")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x_pos, lasso_coefficients, color="tomato", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=45, fontsize=8)
    ax2.set_title("Lasso (L1) Coefficients")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150):
    """Save figure to file.

    Args:
        fig: Matplotlib Figure to save.
        path: Output file path.
        dpi: Resolution.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "from src.visualization import plot_polynomial_fits, plot_error_vs_complexity, plot_bias_variance_tradeoff, plot_regularization_effect; print('Visualization module OK')"
```

Expected: `Visualization module OK`

- [ ] **Step 3: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add visualization module with all plot functions"
```

---

## Task 7: Notebook 01 — Polynomial Regression Demo

**Files:**
- Create: `notebooks/01_polynomial_regression_demo.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/01_polynomial_regression_demo.ipynb` with the following cells:

**Cell 1 (markdown):**
```markdown
# Overfitting & Underfitting: Polynomial Regression Demo

This notebook demonstrates overfitting and underfitting using polynomial curve fitting
on synthetic data generated from `f(x) = sin(2πx)`.
```

**Cell 2 (code):**
```python
import sys
sys.path.insert(0, "..")

import numpy as np
from src.data_generation import generate_polynomial_data, true_function
from src.polynomial_regression import (
    fit_polynomial,
    evaluate_polynomial_degrees,
    compute_bias_variance,
)
from src.visualization import (
    plot_polynomial_fits,
    plot_error_vs_complexity,
    plot_bias_variance_tradeoff,
    save_figure,
)
```

**Cell 3 (markdown):**
```markdown
## 1. Generate Data

We sample 30 points from `f(x) = sin(2πx)` with Gaussian noise (σ = 0.3).
```

**Cell 4 (code):**
```python
X, y = generate_polynomial_data(n_samples=30, noise_std=0.3, seed=42)
print(f"Data shape: X={X.shape}, y={y.shape}")
```

**Cell 5 (markdown):**
```markdown
## 2. Polynomial Fits: Underfitting vs. Good Fit vs. Overfitting

- **Degree 1** (underfitting): Too simple, cannot capture the sinusoidal pattern
- **Degree 4** (good fit): Captures the pattern without memorizing noise
- **Degree 15** (overfitting): Memorizes noise, wild oscillations between data points
```

**Cell 6 (code):**
```python
demo_degrees = [1, 4, 15]
coeffs_dict = {}
for d in demo_degrees:
    coeffs_dict[d] = fit_polynomial(X, y, degree=d)

fig = plot_polynomial_fits(X, y, coeffs_dict, true_fn=true_function)
save_figure(fig, "../report/figures/polynomial_fits.png")
fig  # display in notebook
```

**Cell 7 (markdown):**
```markdown
## 3. Training Error vs. Test Error

As model complexity increases:
- Training error monotonically decreases (the model fits training data better)
- Test error first decreases then increases (overfitting kicks in)
```

**Cell 8 (code):**
```python
X_test, y_test = generate_polynomial_data(n_samples=100, noise_std=0.0, seed=0)
degrees = list(range(1, 16))
results = evaluate_polynomial_degrees(X, y, X_test, y_test, degrees)

fig = plot_error_vs_complexity(results["degrees"], results["train_errors"], results["test_errors"])
save_figure(fig, "../report/figures/error_vs_complexity.png")
fig
```

**Cell 9 (markdown):**
```markdown
## 4. Bias-Variance Decomposition

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

We estimate Bias² and Variance via Monte Carlo: generate 200 datasets,
fit a polynomial on each, and measure the spread of predictions.
```

**Cell 10 (code):**
```python
bv_results = compute_bias_variance(
    true_fn=true_function,
    n_samples=20,
    noise_std=0.3,
    degrees=[1, 2, 3, 4, 5, 7, 9, 12],
    n_repeats=200,
    seed=42,
)

fig = plot_bias_variance_tradeoff(
    bv_results["degrees"],
    bv_results["bias_squared"],
    bv_results["variance"],
    bv_results["total_error"],
)
save_figure(fig, "../report/figures/bias_variance_tradeoff.png")
fig
```

- [ ] **Step 2: Run notebook to verify**

```bash
uv run jupyter execute notebooks/01_polynomial_regression_demo.ipynb
```

Expected: Notebook runs without errors, figures saved to `report/figures/`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_polynomial_regression_demo.ipynb report/figures/
git commit -m "feat: add polynomial regression demo notebook with visualizations"
```

---

## Task 8: Notebook 02 — Regularization Demo

**Files:**
- Create: `notebooks/02_regularization_demo.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/02_regularization_demo.ipynb` with the following cells:

**Cell 1 (markdown):**
```markdown
# Regularization Demo: Ridge (L2) & Lasso (L1)

This notebook demonstrates how regularization combats overfitting
by penalizing large coefficients.

**Mathematical formulation:**
- Ridge (L2): $\min_w \|y - Xw\|^2 + \alpha \|w\|_2^2$
- Lasso (L1): $\min_w \|y - Xw\|^2 + \alpha \|w\|_1$
```

**Cell 2 (code):**
```python
import sys
sys.path.insert(0, "..")

import numpy as np
from src.data_generation import generate_polynomial_data, true_function
from src.regularization import fit_ridge, fit_lasso, evaluate_regularization
from src.visualization import (
    plot_regularization_effect,
    plot_ridge_vs_lasso_coefficients,
    save_figure,
)
```

**Cell 3 (markdown):**
```markdown
## 1. Setup: Overfit Scenario

We use degree=10 polynomial features on 30 data points — a recipe for overfitting.
Regularization will control the model complexity.
```

**Cell 4 (code):**
```python
X_train, y_train = generate_polynomial_data(n_samples=30, noise_std=0.3, seed=42)
X_test, y_test = generate_polynomial_data(n_samples=100, noise_std=0.0, seed=0)

# Reshape for sklearn
X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)
```

**Cell 5 (markdown):**
```markdown
## 2. Ridge Regression: Effect of α

As α increases, coefficients shrink toward zero, reducing overfitting.
But too much regularization leads to underfitting.
```

**Cell 6 (code):**
```python
alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
ridge_results = evaluate_regularization(
    X_train_2d, y_train, X_test_2d, y_test,
    degree=10, alphas=alphas, method="ridge",
)

fig = plot_regularization_effect(
    ridge_results["alphas"],
    ridge_results["train_errors"],
    ridge_results["test_errors"],
    ridge_results["coefficient_norms"],
    method="Ridge",
)
save_figure(fig, "../report/figures/ridge_regularization.png")
fig
```

**Cell 7 (markdown):**
```markdown
## 3. Lasso Regression: Sparsity Effect

Lasso drives some coefficients exactly to zero, performing feature selection.
```

**Cell 8 (code):**
```python
lasso_results = evaluate_regularization(
    X_train_2d, y_train, X_test_2d, y_test,
    degree=10, alphas=alphas, method="lasso",
)

fig = plot_regularization_effect(
    lasso_results["alphas"],
    lasso_results["train_errors"],
    lasso_results["test_errors"],
    lasso_results["coefficient_norms"],
    method="Lasso",
)
save_figure(fig, "../report/figures/lasso_regularization.png")
fig
```

**Cell 9 (markdown):**
```markdown
## 4. Ridge vs. Lasso: Coefficient Comparison

At a moderate α, Ridge shrinks all coefficients evenly, while Lasso zeros out irrelevant ones.
```

**Cell 10 (code):**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly.fit_transform(X_train_2d)

ridge_fit = fit_ridge(X_train_poly, y_train, alpha=0.01)
lasso_fit = fit_lasso(X_train_poly, y_train, alpha=0.01)

feature_names = [f"x^{i+1}" for i in range(10)]
fig = plot_ridge_vs_lasso_coefficients(
    ridge_fit["coefficients"],
    lasso_fit["coefficients"],
    feature_names=feature_names,
)
save_figure(fig, "../report/figures/ridge_vs_lasso_coefficients.png")
fig
```

- [ ] **Step 2: Run notebook to verify**

```bash
uv run jupyter execute notebooks/02_regularization_demo.ipynb
```

Expected: Notebook runs without errors, figures saved.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_regularization_demo.ipynb report/figures/
git commit -m "feat: add regularization demo notebook with Ridge and Lasso"
```

---

## Task 9: LaTeX Report Structure

**Files:**
- Create: `report/main.tex`
- Create: `report/references.bib`

- [ ] **Step 1: Create report/main.tex**

```latex
\documentclass[12pt, a4paper]{article}

% -- Packages --
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{natbib}
\usepackage{float}

\title{Overfitting and Underfitting in Machine Learning}
\author{
  Member A \and Member B \and Member C \\
  \small AIAA 2711 — 2025 Spring
}
\date{}

\begin{document}

\maketitle

\begin{abstract}
This report explores the fundamental concepts of overfitting and underfitting in
machine learning. We derive the bias-variance decomposition, demonstrate these
phenomena through polynomial regression experiments, and analyze regularization
techniques (Ridge and Lasso) as solutions. Applications in deep learning are discussed.
\end{abstract}

% ============================================
\section{Introduction}
\label{sec:introduction}

% Background: why generalization matters in ML
% Problem definition: overfitting vs. underfitting
% Report objective and structure overview

% ============================================
\section{Theoretical Foundation}
\label{sec:theory}

\subsection{Overfitting and Underfitting: Definitions}

% Formal definitions
% Training error vs. test error

\subsection{Bias-Variance Decomposition}

Given a target $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$,
the expected prediction error for a model $\hat{f}(x)$ trained on dataset $D$ is:

\begin{align}
E_D\left[(y - \hat{f}(x))^2\right]
&= \underbrace{\left(f(x) - E_D[\hat{f}(x)]\right)^2}_{\text{Bias}^2}
+ \underbrace{E_D\left[(\hat{f}(x) - E_D[\hat{f}(x)])^2\right]}_{\text{Variance}}
+ \underbrace{\sigma^2}_{\text{Irreducible Error}}
\label{eq:bias-variance}
\end{align}

% Full derivation steps here
% Interpretation: underfitting = high bias, overfitting = high variance

\subsection{Model Complexity and the Tradeoff}

% Model complexity vs. bias/variance curve
% Optimal complexity point

% ============================================
\section{Experimental Design and Results}
\label{sec:experiments}

\subsection{Setup}

% Data generation: f(x) = sin(2*pi*x) + noise
% Polynomial regression as the model family

\subsection{Polynomial Fitting: Visual Demonstration}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/polynomial_fits.png}
    \caption{Polynomial fits at degree 1 (underfitting), 4 (good fit), and 15 (overfitting).}
    \label{fig:polynomial-fits}
\end{figure}

\subsection{Training vs. Test Error}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/error_vs_complexity.png}
    \caption{Training and test error as a function of polynomial degree.}
    \label{fig:error-complexity}
\end{figure}

\subsection{Bias-Variance Decomposition: Empirical Verification}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/bias_variance_tradeoff.png}
    \caption{Empirical bias-variance tradeoff estimated via Monte Carlo simulation.}
    \label{fig:bias-variance}
\end{figure}

% ============================================
\section{Solutions: Regularization}
\label{sec:regularization}

\subsection{Ridge Regression (L2 Regularization)}

The Ridge objective adds an L2 penalty:
\begin{equation}
\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|_2^2 + \alpha \|\mathbf{w}\|_2^2
\label{eq:ridge}
\end{equation}

% Closed-form solution: w = (X^T X + alpha I)^{-1} X^T y
% Geometric interpretation

\subsection{Lasso Regression (L1 Regularization)}

The Lasso objective adds an L1 penalty:
\begin{equation}
\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|_2^2 + \alpha \|\mathbf{w}\|_1
\label{eq:lasso}
\end{equation}

% Sparsity property
% Comparison with Ridge

\subsection{Experimental Results}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/ridge_regularization.png}
    \caption{Ridge regression: error and coefficient norm vs. regularization strength.}
    \label{fig:ridge}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/lasso_regularization.png}
    \caption{Lasso regression: error and coefficient norm vs. regularization strength.}
    \label{fig:lasso}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/ridge_vs_lasso_coefficients.png}
    \caption{Ridge vs. Lasso: coefficient comparison at $\alpha = 0.01$.}
    \label{fig:ridge-vs-lasso}
\end{figure}

\subsection{Cross-Validation}

% K-fold cross-validation principle
% How it helps select optimal model complexity / alpha

\subsection{Other Techniques}

% Early stopping (brief)
% Dropout in neural networks (brief)

% ============================================
\section{Application in AI}
\label{sec:application}

% Overfitting in deep learning
% Relationship between network depth/width and overfitting
% Real-world examples

% ============================================
\section{Conclusion}
\label{sec:conclusion}

% Summary of key findings
% Practical guidelines for managing overfitting/underfitting

% ============================================
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

- [ ] **Step 2: Create report/references.bib**

```bibtex
@book{bishop2006pattern,
  title={Pattern Recognition and Machine Learning},
  author={Bishop, Christopher M.},
  year={2006},
  publisher={Springer}
}

@book{hastie2009elements,
  title={The Elements of Statistical Learning: Data Mining, Inference, and Prediction},
  author={Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome},
  edition={2nd},
  year={2009},
  publisher={Springer}
}

@book{goodfellow2016deep,
  title={Deep Learning},
  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  year={2016},
  publisher={MIT Press}
}

@article{srivastava2014dropout,
  title={Dropout: A Simple Way to Prevent Neural Networks from Overfitting},
  author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  journal={Journal of Machine Learning Research},
  volume={15},
  number={56},
  pages={1929--1958},
  year={2014}
}

@article{tibshirani1996regression,
  title={Regression Shrinkage and Selection via the Lasso},
  author={Tibshirani, Robert},
  journal={Journal of the Royal Statistical Society: Series B},
  volume={58},
  number={1},
  pages={267--288},
  year={1996}
}

@article{hoerl1970ridge,
  title={Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  author={Hoerl, Arthur E. and Kennard, Robert W.},
  journal={Technometrics},
  volume={12},
  number={1},
  pages={55--67},
  year={1970}
}

@inproceedings{ying2019overview,
  title={An Overview of Overfitting and its Solutions},
  author={Ying, Xue},
  booktitle={Journal of Physics: Conference Series},
  volume={1168},
  pages={022022},
  year={2019},
  organization={IOP Publishing}
}
```

- [ ] **Step 3: Commit**

```bash
git add report/main.tex report/references.bib
git commit -m "feat: add LaTeX report structure with figures and references"
```

---

## Task 10: Run All Tests & Final Verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (16 tests).

- [ ] **Step 2: Run all notebooks end-to-end**

```bash
uv run jupyter execute notebooks/01_polynomial_regression_demo.ipynb
uv run jupyter execute notebooks/02_regularization_demo.ipynb
```

Expected: Both run without errors, all figures generated in `report/figures/`.

- [ ] **Step 3: Verify figures exist**

```bash
ls report/figures/
```

Expected files:
- `polynomial_fits.png`
- `error_vs_complexity.png`
- `bias_variance_tradeoff.png`
- `ridge_regularization.png`
- `lasso_regularization.png`
- `ridge_vs_lasso_coefficients.png`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, notebooks execute, figures generated"
```
