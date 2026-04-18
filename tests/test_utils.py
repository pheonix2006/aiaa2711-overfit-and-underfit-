import numpy as np
from src.utils import mse, train_test_split_data


def test_mse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_mse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
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
