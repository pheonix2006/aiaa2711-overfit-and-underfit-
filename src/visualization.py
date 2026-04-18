"""Visualization functions for overfitting/underfitting demonstrations.

All functions accept data/results and return matplotlib Figure objects.
Figures can be displayed in notebooks or saved to report/figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "legend.fontsize": 10,
}


def apply_style():
    """Apply consistent plot style."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_polynomial_fits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coefficients_by_degree: dict[int, np.ndarray],
    true_fn=None,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> Figure:
    """Plot data points and polynomial fits for multiple degrees side by side."""
    apply_style()
    n_plots = len(coefficients_by_degree)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    x_smooth = np.linspace(x_range[0], x_range[1], 200)

    for idx, (degree, coeffs) in enumerate(coefficients_by_degree.items()):
        ax = axes[0, idx]
        ax.scatter(X_train, y_train, c="steelblue", s=30, alpha=0.7, label="Training data", zorder=3)
        if true_fn is not None:
            ax.plot(x_smooth, true_fn(x_smooth), "g--", alpha=0.6, label="True function")
        y_fit = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_fit, "r-", label=f"Degree {degree}")
        ax.set_title(f"Polynomial Degree = {degree}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-2, 2)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_error_vs_complexity(
    degrees: list[int],
    train_errors: list[float],
    test_errors: list[float],
) -> Figure:
    """Plot train and test error as a function of model complexity (degree)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, train_errors, "o-", color="steelblue", label="Training Error")
    ax.plot(degrees, test_errors, "s-", color="tomato", label="Test Error")
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Training vs. Test Error")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bias_variance_tradeoff(
    degrees: list[int],
    bias_squared: list[float],
    variance: list[float],
    total_error: list[float],
) -> Figure:
    """Plot bias^2, variance, and total error vs. model complexity."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, bias_squared, "o-", color="steelblue", label="Bias²")
    ax.plot(degrees, variance, "s-", color="tomato", label="Variance")
    ax.plot(degrees, total_error, "^-", color="seagreen", label="Total Error (Bias² + Var + σ²)")
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("Error")
    ax.set_title("Bias-Variance Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mid = len(degrees) // 2
    ax.axvspan(degrees[0], degrees[mid], alpha=0.05, color="blue")
    ax.axvspan(degrees[mid], degrees[-1], alpha=0.05, color="red")
    ax.text(degrees[1], max(total_error) * 0.9, "Underfitting\n(High Bias)", fontsize=9, color="blue")
    ax.text(degrees[-2], max(total_error) * 0.9, "Overfitting\n(High Variance)", fontsize=9, color="red", ha="right")
    fig.tight_layout()
    return fig


def plot_regularization_effect(
    alphas: list[float],
    train_errors: list[float],
    test_errors: list[float],
    coefficient_norms: list[float],
    method: str = "Ridge",
) -> Figure:
    """Plot regularization effect: error and coefficient norms vs. alpha."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(alphas, train_errors, "o-", color="steelblue", label="Training Error")
    ax1.plot(alphas, test_errors, "s-", color="tomato", label="Test Error")
    ax1.set_xscale("log")
    ax1.set_xlabel(f"{method} α (Regularization Strength)")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title(f"{method}: Error vs. Regularization Strength")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(alphas, coefficient_norms, "D-", color="mediumpurple")
    ax2.set_xscale("log")
    ax2.set_xlabel(f"{method} α (Regularization Strength)")
    ax2.set_ylabel("‖w‖₂ (Coefficient Norm)")
    ax2.set_title(f"{method}: Coefficient Shrinkage")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ridge_vs_lasso_coefficients(
    ridge_coefficients: np.ndarray,
    lasso_coefficients: np.ndarray,
    feature_names: list[str] | None = None,
) -> Figure:
    """Side-by-side comparison of Ridge and Lasso coefficient values."""
    apply_style()
    n = len(ridge_coefficients)
    if feature_names is None:
        feature_names = [f"x^{i+1}" for i in range(n)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x_pos = np.arange(n)
    ax1.bar(x_pos, ridge_coefficients, color="steelblue", alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, fontsize=8)
    ax1.set_title("Ridge (L2) Coefficients")
    ax1.set_ylabel("Coefficient Value")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(x_pos, lasso_coefficients, color="tomato", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=45, fontsize=8)
    ax2.set_title("Lasso (L1) Coefficients")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150):
    """Save figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dl_synthetic_fits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    predictions_by_arch: dict[str, np.ndarray],
    x_smooth: np.ndarray,
    true_fn=None,
) -> Figure:
    """Plot MLP regression fits for different architectures.

    Args:
        X_train: Training x values (n,).
        y_train: Training y values (n,).
        predictions_by_arch: {"label": y_pred_on_x_smooth} dict.
        x_smooth: Dense x grid for plotting smooth curves.
        true_fn: Optional ground-truth function.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    n_plots = len(predictions_by_arch)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)

    for idx, (label, y_pred) in enumerate(predictions_by_arch.items()):
        ax = axes[0, idx]
        ax.scatter(X_train, y_train, c="steelblue", s=30, alpha=0.7, label="Training data", zorder=3)
        if true_fn is not None:
            ax.plot(x_smooth, true_fn(x_smooth), "g--", alpha=0.6, label="True function")
        ax.plot(x_smooth, y_pred, "r-", label=f"MLP {label}")
        ax.set_title(f"MLP: {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-2, 2)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    best_epoch: int | None = None,
    title: str = "Training Curves",
) -> Figure:
    """Plot train/test loss over epochs with optional early stopping marker.

    Args:
        train_losses: Training loss per epoch.
        test_losses: Test loss per epoch.
        best_epoch: Epoch index for early stopping marker (None = no marker).
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "-", color="steelblue", label="Training Loss")
    ax.plot(epochs, test_losses, "-", color="tomato", label="Test Loss")

    if best_epoch is not None:
        ax.axvline(x=best_epoch + 1, color="green", linestyle="--", alpha=0.7, label=f"Early Stop (epoch {best_epoch + 1})")
        ax.plot(best_epoch + 1, test_losses[best_epoch], "g*", markersize=15, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dropout_comparison(
    history_no_dropout: dict,
    history_with_dropout: dict,
    title: str = "Effect of Dropout on MNIST Classification",
) -> Figure:
    """Side-by-side comparison of train/test loss with and without dropout.

    Args:
        history_no_dropout: Dict with train_losses, test_losses.
        history_with_dropout: Dict with train_losses, test_losses.
        title: Overall figure title.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs1 = range(1, len(history_no_dropout["train_losses"]) + 1)
    ax1.plot(epochs1, history_no_dropout["train_losses"], "-", color="steelblue", label="Train Loss")
    ax1.plot(epochs1, history_no_dropout["test_losses"], "-", color="tomato", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Without Dropout")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    epochs2 = range(1, len(history_with_dropout["train_losses"]) + 1)
    ax2.plot(epochs2, history_with_dropout["train_losses"], "-", color="steelblue", label="Train Loss")
    ax2.plot(epochs2, history_with_dropout["test_losses"], "-", color="tomato", label="Test Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("With Dropout (p=0.5)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_network_complexity_comparison(
    param_counts: list[int],
    train_errors: list[float],
    test_errors: list[float],
    arch_labels: list[str] | None = None,
) -> Figure:
    """Plot test error vs. network parameter count.

    Args:
        param_counts: Number of parameters per architecture.
        train_errors: Final train loss per architecture.
        test_errors: Final test loss per architecture.
        arch_labels: Optional labels for each architecture.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_counts, train_errors, "o-", color="steelblue", label="Training Error")
    ax.plot(param_counts, test_errors, "s-", color="tomato", label="Test Error")

    if arch_labels is not None:
        for i, label in enumerate(arch_labels):
            ax.annotate(label, (param_counts[i], test_errors[i]),
                       textcoords="offset points", xytext=(0, 10),
                       fontsize=8, ha="center")

    ax.set_xlabel("Number of Parameters (Network Complexity)")
    ax.set_ylabel("Loss")
    ax.set_title("Network Complexity vs. Generalization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
