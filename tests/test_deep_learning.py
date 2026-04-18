"""Tests for deep learning module."""

import numpy as np
import torch

from src.deep_learning import (
    build_mlp,
    train_model,
    evaluate_network_complexity,
    load_mnist_subset,
)


def test_build_mlp_output_shape_regression():
    """MLP with regression output should produce (batch, 1)."""
    model = build_mlp(input_dim=10, hidden_layers=[32, 16], output_dim=1)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 1)


def test_build_mlp_output_shape_classification():
    """MLP with 10-class output should produce (batch, 10)."""
    model = build_mlp(input_dim=784, hidden_layers=[128, 64], output_dim=10)
    x = torch.randn(3, 784)
    out = model(x)
    assert out.shape == (3, 10)


def test_build_mlp_with_dropout():
    """Dropout layers should appear when dropout_rate > 0."""
    model = build_mlp(input_dim=10, hidden_layers=[32], output_dim=1, dropout_rate=0.5)
    layer_types = [type(m).__name__ for m in model.modules()]
    assert "Dropout" in layer_types


def test_train_model_returns_history():
    """train_model should return dict with expected keys."""
    torch.manual_seed(42)
    model = build_mlp(input_dim=1, hidden_layers=[16], output_dim=1)
    X_train = torch.linspace(0, 1, 30).unsqueeze(1)
    y_train = torch.sin(2 * np.pi * X_train)
    X_test = torch.linspace(0, 1, 10).unsqueeze(1)
    y_test = torch.sin(2 * np.pi * X_test)

    result = train_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.01)
    assert "train_losses" in result
    assert "test_losses" in result
    assert "best_epoch" in result
    assert len(result["train_losses"]) == 10
    assert len(result["test_losses"]) == 10


def test_train_model_loss_decreases():
    """Training loss should generally decrease over epochs."""
    torch.manual_seed(42)
    model = build_mlp(input_dim=1, hidden_layers=[32, 16], output_dim=1)
    X_train = torch.linspace(0, 1, 50).unsqueeze(1)
    y_train = torch.sin(2 * np.pi * X_train)
    X_test = X_train.clone()
    y_test = y_train.clone()

    result = train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01)
    assert result["train_losses"][0] > result["train_losses"][-1]


def test_evaluate_network_complexity_structure():
    """evaluate_network_complexity should return dict with correct lengths."""
    torch.manual_seed(42)
    X_train = torch.linspace(0, 1, 30).unsqueeze(1)
    y_train = torch.sin(2 * np.pi * X_train)
    X_test = torch.linspace(0, 1, 10).unsqueeze(1)
    y_test = torch.sin(2 * np.pi * X_test)

    architectures = [[8], [32, 16]]
    result = evaluate_network_complexity(
        X_train, y_train, X_test, y_test,
        architectures=architectures, seed=42, epochs=50,
    )
    assert len(result["architectures"]) == 2
    assert len(result["param_counts"]) == 2
    assert len(result["train_errors"]) == 2
    assert len(result["test_errors"]) == 2


def test_load_mnist_subset_shapes():
    """MNIST subset should have correct shapes."""
    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=100, n_test=50, seed=42,
    )
    assert X_train.shape == (100, 784)
    assert y_train.shape == (100,)
    assert X_test.shape == (50, 784)
    assert y_test.shape == (50,)
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
