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
    assert np.mean(np.abs(y - expected)) > 0.1
