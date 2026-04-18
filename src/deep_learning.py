"""Deep learning module: MLP building, training, and evaluation for overfitting demos."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def build_mlp(
    input_dim: int,
    hidden_layers: list[int],
    output_dim: int = 1,
    dropout_rate: float = 0.0,
    activation: str = "relu",
) -> nn.Module:
    """Build a configurable Multi-Layer Perceptron.

    Args:
        input_dim: Number of input features.
        hidden_layers: List of hidden layer sizes, e.g. [128, 64].
        output_dim: Number of output units (1 for regression, N for N-class).
        dropout_rate: Dropout probability (0.0 = no dropout).
        activation: Activation function ("relu" or "tanh").

    Returns:
        nn.Sequential model.
    """
    act_fn = nn.ReLU if activation == "relu" else nn.Tanh
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev_dim = h

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.01,
    task: str = "regression",
    patience: int = 0,
    batch_size: int = 0,
) -> dict:
    """Train the model and record train/test loss history.

    Args:
        model: PyTorch model to train.
        X_train, y_train: Training data as tensors.
        X_test, y_test: Test data as tensors.
        epochs: Number of training epochs.
        lr: Learning rate.
        task: "regression" (MSELoss) or "classification" (CrossEntropyLoss).
        patience: Early stopping patience (0 = disabled).
        batch_size: Mini-batch size (0 = full batch).

    Returns:
        Dict with keys: train_losses, test_losses, best_epoch, model.
    """
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    use_batches = batch_size > 0 and batch_size < len(X_train)
    if use_batches:
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        if use_batches:
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_losses.append(epoch_loss / len(X_train))
        else:
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        test_losses.append(test_loss)

        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            break

    # Restore best model if early stopping was used
    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "best_epoch": best_epoch,
        "model": model,
    }


def evaluate_network_complexity(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    architectures: list[list[int]],
    seed: int = 42,
    epochs: int = 300,
    lr: float = 0.01,
    task: str = "regression",
) -> dict:
    """Compare different network architectures on the same data.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        architectures: List of hidden layer configs, e.g. [[8], [32, 16], [128, 64, 32]].
        seed: Random seed for reproducibility.
        epochs: Training epochs per architecture.
        lr: Learning rate.
        task: "regression" or "classification".

    Returns:
        Dict with keys: architectures, param_counts, train_errors, test_errors, histories.
    """
    input_dim = X_train.shape[1]
    output_dim = 1 if task == "regression" else len(torch.unique(y_train))

    param_counts = []
    train_errors = []
    test_errors = []
    histories = []

    for arch in architectures:
        torch.manual_seed(seed)
        model = build_mlp(input_dim, arch, output_dim)
        n_params = sum(p.numel() for p in model.parameters())
        param_counts.append(n_params)

        result = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, task=task,
        )
        train_errors.append(result["train_losses"][-1])
        test_errors.append(result["test_losses"][-1])
        histories.append(result)

    return {
        "architectures": architectures,
        "param_counts": param_counts,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "histories": histories,
    }


def load_mnist_subset(
    n_train: int = 500,
    n_test: int = 1000,
    seed: int = 42,
    data_dir: str = "data",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST dataset, flatten and normalize, return subset as tensors.

    Downloads MNIST automatically to data_dir/ if not present.

    Args:
        n_train: Number of training samples to use.
        n_test: Number of test samples to use.
        seed: Random seed for subset selection.
        data_dir: Directory to download/cache MNIST data.

    Returns:
        (X_train, y_train, X_test, y_test) as float32/int64 tensors.
        X shape: (n, 784), y shape: (n,).
    """
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    data_path = Path(data_dir)

    train_dataset = datasets.MNIST(root=str(data_path), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_path), train=False, download=True, transform=transform)

    # Convert to tensors and flatten
    X_train_full = train_dataset.data.float().reshape(-1, 784) / 255.0
    y_train_full = train_dataset.targets.long()
    X_test_full = test_dataset.data.float().reshape(-1, 784) / 255.0
    y_test_full = test_dataset.targets.long()

    # Select random subset
    rng = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(X_train_full), generator=rng)[:n_train]
    test_idx = torch.randperm(len(X_test_full), generator=rng)[:n_test]

    return (
        X_train_full[train_idx],
        y_train_full[train_idx],
        X_test_full[test_idx],
        y_test_full[test_idx],
    )
