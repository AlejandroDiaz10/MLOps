"""
Plotting and visualization utilities for model evaluation.
Generates professional charts for reports and presentations.
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from loguru import logger

from fase2.config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(
    cm: np.ndarray, model_name: str, save_path: Optional[Path] = None
) -> Path:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        model_name: Name of the model
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=2,
        linecolor="black",
        ax=ax,
        vmin=0,
    )

    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xticklabels(["Bad Credit (0)", "Good Credit (1)"])
    ax.set_yticklabels(["Bad Credit (0)", "Good Credit (1)"])

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    output_path = save_path / f"confusion_matrix_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Confusion matrix saved: {output_path}")
    return output_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve
    ax.plot(
        fpr, tpr, color="darkorange", lw=3, label=f"{model_name} (AUC = {roc_auc:.4f})"
    )

    # Plot diagonal reference line
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier (AUC = 0.5000)",
    )

    # Highlight threshold area
    threshold_met = roc_auc >= config.model.auc_threshold
    color = "green" if threshold_met else "red"
    ax.axhline(
        y=config.model.auc_threshold,
        color=color,
        linestyle=":",
        alpha=0.5,
        label=f"Threshold ({config.model.auc_threshold})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(f"ROC Curve - {model_name}", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    output_path = save_path / f"roc_curve_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ ROC curve saved: {output_path}")
    return output_path


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        model_name: Name of the model
        top_n: Number of top features to show
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    # Get top N features
    top_features = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(
        range(len(top_features)),
        top_features["Importance"],
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Feature Importances - {model_name}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features["Importance"])):
        ax.text(val, i, f"  {val:.4f}", va="center", fontweight="bold")

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    output_path = save_path / f"feature_importance_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Feature importance plot saved: {output_path}")
    return output_path


def plot_model_comparison(
    results: Dict[str, Dict], save_path: Optional[Path] = None
) -> Path:
    """
    Plot comparison of multiple models across metrics.

    Args:
        results: Dictionary with model results {model_name: {'metrics': {...}}}
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    model_names = []
    metrics_data = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "AUC-ROC": [],
    }

    for model_name, result in results.items():
        if "error" in result:
            continue

        model_names.append(model_name)
        m = result["metrics"]
        metrics_data["Accuracy"].append(m["accuracy"])
        metrics_data["Precision"].append(m["precision"])
        metrics_data["Recall"].append(m["recall"])
        metrics_data["F1-Score"].append(m["f1_score"])
        metrics_data["AUC-ROC"].append(m.get("auc_roc", 0))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar chart of all metrics
    x = np.arange(len(model_names))
    width = 0.15

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (metric_name, values) in enumerate(metrics_data.items()):
        offset = width * (i - 2)
        axes[0].bar(
            x + offset,
            values,
            width,
            label=metric_name,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
        )

    axes[0].set_xlabel("Model", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Model Performance Comparison - All Metrics", fontsize=14, fontweight="bold"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim([0, 1.05])

    # Add threshold line
    axes[0].axhline(
        y=config.model.auc_threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=2,
        label=f"AUC Threshold ({config.model.auc_threshold})",
    )

    # Plot 2: AUC-ROC focused comparison
    colors_auc = [
        "green" if auc >= config.model.auc_threshold else "red"
        for auc in metrics_data["AUC-ROC"]
    ]

    bars = axes[1].bar(
        model_names,
        metrics_data["AUC-ROC"],
        color=colors_auc,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    axes[1].set_xlabel("Model", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("AUC-ROC Score", fontsize=12, fontweight="bold")
    axes[1].set_title("AUC-ROC Comparison", fontsize=14, fontweight="bold")
    axes[1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1].axhline(
        y=config.model.auc_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Threshold ({config.model.auc_threshold})",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim([0, 1.05])

    # Add value labels on bars
    for bar, value in zip(bars, metrics_data["AUC-ROC"]):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()

    output_path = save_path / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Model comparison plot saved: {output_path}")
    return output_path
