import numpy as np
from src.polynomial_regression import (
    fit_polynomial,
    predict_polynomial,
    evaluate_polynomial_degrees,
    compute_bias_variance,
)


def test_fit_polynomial_linear():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * X + 1.0
    coeffs = fit_polynomial(X, y, degree=1)
    assert len(coeffs) == 2


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
    rng = np.random.default_rng(42)
    X_train = np.linspace(0, 1, 15)
    y_train = np.sin(2 * np.pi * X_train) + rng.normal(0, 0.3, 15)
    X_test = np.linspace(0, 1, 50)
    y_test = np.sin(2 * np.pi * X_test)
    results = evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, [1, 4, 14])
    assert results["train_errors"][2] < results["train_errors"][0]
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
    result = compute_bias_variance(
        true_fn=lambda x: np.sin(2 * np.pi * x),
        n_samples=20,
        noise_std=0.3,
        degrees=[1, 9],
        n_repeats=100,
        seed=42,
    )
    assert result["bias_squared"][0] > result["bias_squared"][1]
    assert result["variance"][1] > result["variance"][0]
