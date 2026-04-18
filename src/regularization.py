"""Regularization: Ridge (L2) and Lasso (L1) regression."""

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

from src.utils import mse


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Fit Ridge (L2) regression.

    Args:
        X: Feature matrix (n_samples, n_features).
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
