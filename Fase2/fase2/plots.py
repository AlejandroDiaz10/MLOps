"""
Plotting and visualization utilities for model evaluation.
Generates professional charts for reports and presentations.
"""

from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from loguru import logger

from fase2.config import config

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def plot_target_distribution(
    df: pd.DataFrame, target_col: str = None, save_path: Optional[Path] = None
) -> Path:
    """
    Plot target variable distribution.

    Args:
        df: DataFrame with target column
        target_col: Name of target column
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if target_col is None:
        target_col = config.data.target_col

    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    counts = df[target_col].value_counts()
    colors = ["#d62728", "#2ca02c"]  # Red for 0, Green for 1

    bars = ax.bar(
        counts.index, counts.values, color=colors, alpha=0.8, edgecolor="black"
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({100*height/len(df):.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_xlabel("Credit Risk", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title("Target Variable Distribution", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Bad Credit (0)", "Good Credit (1)"])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = save_path / "target_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Target distribution plot saved: {output_path}")
    return output_path


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


def plot_multiple_roc_curves(
    results: Dict[str, Dict], y_test: np.ndarray, save_path: Optional[Path] = None
) -> Path:
    """
    Plot ROC curves for multiple models on the same chart.

    Args:
        results: Dictionary with model results
        y_test: True test labels
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (model_name, result) in enumerate(results.items()):
        if "error" in result or "y_proba" not in result:
            continue

        y_proba = result["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=3,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )

    # Plot diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=2,
        linestyle="--",
        label="Random Classifier (AUC = 0.5000)",
        alpha=0.5,
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(
        "ROC Curves - Model Comparison", fontsize=16, fontweight="bold", pad=20
    )
    ax.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = save_path / "roc_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Multiple ROC curves saved: {output_path}")
    return output_path


def plot_training_history(
    results: Dict[str, Dict], save_path: Optional[Path] = None
) -> Path:
    """
    Plot cross-validation scores for all models.

    Args:
        results: Dictionary with model results including CV scores
        save_path: Directory to save plot

    Returns:
        Path to saved figure
    """
    if save_path is None:
        save_path = config.paths.figures_dir

    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    model_names = []
    cv_means = []
    cv_stds = []

    for model_name, result in results.items():
        if "error" in result:
            continue

        # Get CV scores from metadata if available
        if "cv_scores" in result:
            scores = result["cv_scores"]
            model_names.append(model_name)
            cv_means.append(np.mean(scores))
            cv_stds.append(np.std(scores))

    if not model_names:
        logger.warning("No CV scores available for plotting")
        return None

    x = np.arange(len(model_names))

    colors = [
        "#2ca02c" if mean >= config.model.auc_threshold else "#d62728"
        for mean in cv_means
    ]

    bars = ax.bar(
        x,
        cv_means,
        yerr=cv_stds,
        capsize=10,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cross-Validation AUC Score", fontsize=14, fontweight="bold")
    ax.set_title("Cross-Validation Performance", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.axhline(
        y=config.model.auc_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Threshold ({config.model.auc_threshold})",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.4f}±{std:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()

    output_path = save_path / "cv_scores_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ CV scores plot saved: {output_path}")
    return output_path


def generate_all_plots(
    df: pd.DataFrame = None,
    results: Dict[str, Dict] = None,
    y_test: np.ndarray = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generate all visualization plots.

    Args:
        df: DataFrame with data (for target distribution)
        results: Model comparison results
        y_test: True test labels
        save_path: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    logger.info("=" * 70)
    logger.info("GENERATING ALL VISUALIZATIONS")
    logger.info("=" * 70)

    plots = {}

    # Target distribution
    if df is not None:
        try:
            plots["target_distribution"] = plot_target_distribution(
                df, save_path=save_path
            )
        except Exception as e:
            logger.error(f"Failed to plot target distribution: {e}")

    # Model comparison plots
    if results is not None:
        try:
            plots["model_comparison"] = plot_model_comparison(
                results, save_path=save_path
            )
        except Exception as e:
            logger.error(f"Failed to plot model comparison: {e}")

        # Multiple ROC curves
        if y_test is not None:
            try:
                plots["roc_comparison"] = plot_multiple_roc_curves(
                    results, y_test, save_path=save_path
                )
            except Exception as e:
                logger.error(f"Failed to plot ROC comparison: {e}")

        # CV scores
        try:
            cv_plot = plot_training_history(results, save_path=save_path)
            if cv_plot:
                plots["cv_scores"] = cv_plot
        except Exception as e:
            logger.error(f"Failed to plot CV scores: {e}")

        # Individual model plots
        for model_name, result in results.items():
            if "error" in result:
                continue

            try:
                # Confusion matrix
                if "y_pred" in result and y_test is not None:
                    cm = confusion_matrix(y_test, result["y_pred"])
                    plots[f"cm_{model_name}"] = plot_confusion_matrix(
                        cm, model_name, save_path
                    )

                # ROC curve
                if "y_proba" in result and y_test is not None:
                    plots[f"roc_{model_name}"] = plot_roc_curve(
                        y_test, result["y_proba"], model_name, save_path
                    )

                # Feature importance
                if "feature_importance" in result:
                    plots[f"fi_{model_name}"] = plot_feature_importance(
                        result["feature_importance"], model_name, save_path=save_path
                    )

            except Exception as e:
                logger.error(f"Failed to plot {model_name}: {e}")

    logger.success(f"✓ Generated {len(plots)} visualization plots")
    return plots
