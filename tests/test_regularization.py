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
