"""
Visualization utilities for EDA and model evaluation
"""

from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

from fase2.config import FIGURES_DIR, PROCESSED_DATA_DIR, TARGET_COL

app = typer.Typer()

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ============= FUNCIONES REUTILIZABLES =============


def plot_target_distribution(df: pd.DataFrame, save_path: Path = FIGURES_DIR):
    """Plot target variable distribution"""
    logger.info(f"Plotting target distribution for '{TARGET_COL}'...")

    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    value_counts = df[TARGET_COL].value_counts().sort_index()
    colors = ["#d62728", "#2ca02c"]
    axes[0].bar(value_counts.index, value_counts.values, color=colors, alpha=0.8)
    axes[0].set_title("Credit Risk Distribution", fontweight="bold", fontsize=14)
    axes[0].set_xlabel("Credit Risk")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Bad (0)", "Good (1)"])

    # Add count labels
    for i, (idx, val) in enumerate(value_counts.items()):
        axes[0].text(i, val + 5, str(val), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        value_counts,
        labels=["Bad (0)", "Good (1)"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Credit Risk Proportion", fontweight="bold", fontsize=14)

    plt.tight_layout()
    output_file = save_path / "target_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"✓ Target distribution saved to: {output_file}")
    return output_file


def plot_confusion_matrix(
    cm: np.ndarray, model_name: str, save_path: Path = FIGURES_DIR
):
    """Plot confusion matrix heatmap"""
    logger.info(f"Plotting confusion matrix for {model_name}...")

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad (0)", "Good (1)"],
        yticklabels=["Bad (0)", "Good (1)"],
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = 100 * cm[i, j] / total
            ax.text(
                j + 0.5,
                i + 0.85,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                color="gray",
                fontsize=9,
            )

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    output_file = save_path / f"confusion_matrix_{safe_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"✓ Confusion matrix saved to: {output_file}")
    return output_file


def plot_roc_curve(y_true, y_score, model_name: str, save_path: Path = FIGURES_DIR):
    """Plot ROC curve"""
    logger.info(f"Plotting ROC curve for {model_name}...")

    save_path.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier (AUC = 0.50)",
    )

    # Mark AUC threshold
    if roc_auc >= 0.75:
        ax.axhline(
            y=0.75, color="green", linestyle=":", alpha=0.5, label="Target AUC (0.75)"
        )

    # Optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        markersize=10,
        label=f"Optimal threshold ({optimal_threshold:.2f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    output_file = save_path / f"roc_curve_{safe_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"✓ ROC curve saved to: {output_file}")
    return output_file


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 15,
    save_path: Path = FIGURES_DIR,
):
    """Plot feature importance for tree-based models"""
    logger.info(f"Plotting feature importance for {model_name}...")

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)

    ax.barh(
        range(len(top_features)),
        top_features["Importance"],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances - {model_name}",
        fontweight="bold",
        fontsize=14,
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_file = save_path / "feature_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"✓ Feature importance saved to: {output_file}")
    return output_file


# ============= CLI COMMAND =============


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "X_train.csv",
    output_dir: Path = FIGURES_DIR,
):
    """
    Generate exploratory data analysis plots.

    Creates basic visualizations from processed data.
    """
    logger.info("=" * 60)
    logger.info("STARTING PLOT GENERATION")
    logger.info("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: Plot a simple histogram
    logger.info(f"Loading data from: {input_path}")

    if input_path.exists():
        df = pd.read_csv(input_path)
        logger.info(f"Data shape: {df.shape}")

        # Generate a sample plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df.iloc[:, 0].hist(bins=30, ax=ax, color="steelblue", alpha=0.7)
        ax.set_title("Sample Feature Distribution", fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        output_file = output_dir / "sample_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.success(f"✓ Sample plot saved to: {output_file}")
    else:
        logger.warning(f"⚠ Input file not found: {input_path}")
        logger.info("Run 'python -m fase2.features' first to generate processed data")

    logger.success("=" * 60)
    logger.success("✓ PLOT GENERATION COMPLETE")
    logger.success("=" * 60)


if __name__ == "__main__":
    app()
