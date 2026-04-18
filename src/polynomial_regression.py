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

        y_range = np.abs(true_fn(x_test)).max() * 10 + 10
        for i in range(n_repeats):
            x_train = np.sort(rng.uniform(x_range[0], x_range[1], n_samples))
            y_train = true_fn(x_train) + rng.normal(0, noise_std, n_samples)
            coeffs = fit_polynomial(x_train, y_train, d)
            pred = predict_polynomial(x_test, coeffs)
            predictions[i] = np.clip(pred, -y_range, y_range)

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
