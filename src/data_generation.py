"""Synthetic data generation for overfitting/underfitting demos."""

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """The ground-truth function: f(x) = sin(2*pi*x).

    Classic function from Bishop's PRML for demonstrating
    polynomial curve fitting and overfitting.
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
