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
    """Split data into train and test sets."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
