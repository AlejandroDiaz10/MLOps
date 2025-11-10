"""
Drift Visualizer - Create plots for drift analysis.

Generates comprehensive visualizations:
- Feature distribution comparisons
- Drift heatmaps
- Performance degradation charts
- Alert dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class DriftVisualizer:
    """
    Create visualizations for data drift analysis.

    Example usage:
        viz = DriftVisualizer(output_dir='reports/figures')
        viz.plot_feature_distributions(X_ref, X_mon, top_n=5)
        viz.plot_drift_heatmap(drift_results)
    """

    def __init__(self, output_dir: str = "reports/figures/drift"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DriftVisualizer initialized, output: {self.output_dir}")

    def plot_feature_distributions(
        self,
        X_reference: pd.DataFrame,
        X_monitoring: pd.DataFrame,
        features: Optional[List[str]] = None,
        top_n: int = 6,
        save_path: Optional[str] = None,
    ):
        """
        Plot distribution comparisons for features.

        Args:
            X_reference: Reference data
            X_monitoring: Monitoring data
            features: Specific features to plot (None = auto-select top N)
            top_n: Number of features to plot
            save_path: Path to save plot
        """
        if features is None:
            features = X_reference.columns[:top_n]
        else:
            features = features[:top_n]

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Plot histograms
            ax.hist(
                X_reference[feature].dropna(),
                bins=30,
                alpha=0.5,
                label="Reference",
                color="blue",
                density=True,
            )
            ax.hist(
                X_monitoring[feature].dropna(),
                bins=30,
                alpha=0.5,
                label="Monitoring",
                color="red",
                density=True,
            )

            ax.set_title(f"{feature}", fontweight="bold")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "feature_distributions.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Saved feature distributions plot: {save_path}")
        plt.close()

    def plot_drift_heatmap(self, drift_results: Dict, save_path: Optional[str] = None):
        """
        Plot heatmap of drift scores across features and tests.

        Args:
            drift_results: Results from DriftDetector.detect_drift()
            save_path: Path to save plot
        """
        # Extract drift scores
        features = list(drift_results["feature_results"].keys())

        # Get unique test types
        test_types = set()
        for feature_tests in drift_results["feature_results"].values():
            test_types.update(feature_tests.keys())
        test_types = sorted(test_types)

        # Create matrix
        matrix = np.zeros((len(features), len(test_types)))

        for i, feature in enumerate(features):
            for j, test_type in enumerate(test_types):
                if test_type in drift_results["feature_results"][feature]:
                    matrix[i, j] = drift_results["feature_results"][feature][
                        test_type
                    ].drift_score

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.3)))

        sns.heatmap(
            matrix,
            xticklabels=[t.upper() for t in test_types],
            yticklabels=features,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={"label": "Drift Score"},
            vmin=0,
            vmax=1,
            ax=ax,
        )

        ax.set_title(
            "Data Drift Heatmap\n(Higher values = More drift)",
            fontweight="bold",
            fontsize=14,
        )
        ax.set_xlabel("Test Type", fontweight="bold")
        ax.set_ylabel("Feature", fontweight="bold")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "drift_heatmap.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Saved drift heatmap: {save_path}")
        plt.close()

    def plot_performance_comparison(
        self, scenarios_results: Dict, save_path: Optional[str] = None
    ):
        """
        Plot performance metrics across scenarios.

        Args:
            scenarios_results: Dict mapping scenario -> PerformanceComparison
            save_path: Path to save plot
        """
        scenarios = list(scenarios_results.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

        # Extract baseline and monitoring values
        baseline_values = {
            metric: scenarios_results[scenarios[0]].baseline.__dict__[metric]
            for metric in metrics
        }

        monitoring_values = {
            scenario: {
                metric: results.monitoring.__dict__[metric] for metric in metrics
            }
            for scenario, results in scenarios_results.items()
        }

        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Plot baseline
            baseline_val = baseline_values[metric]
            ax.axhline(
                y=baseline_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label="Baseline",
                alpha=0.7,
            )

            # Plot monitoring values
            scenario_names = list(monitoring_values.keys())
            values = [monitoring_values[s][metric] for s in scenario_names]

            bars = ax.bar(range(len(scenario_names)), values, alpha=0.7)

            # Color bars based on degradation
            for i, (scenario, bar) in enumerate(zip(scenario_names, bars)):
                degradation = scenarios_results[scenario].degradation[metric]
                if degradation < -0.05:  # 5% degradation
                    bar.set_color("red")
                elif degradation < 0:
                    bar.set_color("orange")
                else:
                    bar.set_color("green")

            ax.set_title(metric.upper().replace("_", " "), fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_xticks(range(len(scenario_names)))
            ax.set_xticklabels(scenario_names, rotation=45, ha="right", fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim([0, 1.05])

        # Hide last subplot (we have 5 metrics, 6 subplots)
        axes[5].set_visible(False)

        plt.suptitle(
            "Performance Metrics Across Drift Scenarios\n"
            "(Green=Baseline, Red=Significant Degradation)",
            fontweight="bold",
            fontsize=14,
        )
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Saved performance comparison: {save_path}")
        plt.close()

    def plot_drift_summary_dashboard(
        self,
        drift_results: Dict,
        performance_results: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Create comprehensive dashboard with drift and performance info.

        Args:
            drift_results: Dict mapping scenario -> drift results
            performance_results: Dict mapping scenario -> performance comparison
            save_path: Path to save plot
        """
        scenarios = list(drift_results.keys())

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Drift percentage per scenario
        ax1 = fig.add_subplot(gs[0, :])
        drift_pcts = [
            drift_results[s]["drift_summary"]["drift_percentage"] for s in scenarios
        ]
        bars = ax1.barh(scenarios, drift_pcts, color="steelblue")

        for i, (bar, pct) in enumerate(zip(bars, drift_pcts)):
            if pct > 50:
                bar.set_color("red")
            elif pct > 25:
                bar.set_color("orange")

        ax1.set_xlabel("% Features Drifted", fontweight="bold")
        ax1.set_title("Data Drift Summary", fontweight="bold", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="x")

        # 2. AUC-ROC comparison
        ax2 = fig.add_subplot(gs[1, 0])
        baseline_auc = performance_results[scenarios[0]].baseline.auc_roc
        monitoring_aucs = [performance_results[s].monitoring.auc_roc for s in scenarios]

        ax2.axhline(
            y=baseline_auc, color="green", linestyle="--", linewidth=2, label="Baseline"
        )
        ax2.plot(
            scenarios,
            monitoring_aucs,
            marker="o",
            color="red",
            linewidth=2,
            label="Monitoring",
        )
        ax2.set_ylabel("AUC-ROC", fontweight="bold")
        ax2.set_title("AUC-ROC Degradation", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # 3. Accuracy comparison
        ax3 = fig.add_subplot(gs[1, 1])
        baseline_acc = performance_results[scenarios[0]].baseline.accuracy
        monitoring_accs = [
            performance_results[s].monitoring.accuracy for s in scenarios
        ]

        ax3.axhline(
            y=baseline_acc, color="green", linestyle="--", linewidth=2, label="Baseline"
        )
        ax3.plot(
            scenarios,
            monitoring_accs,
            marker="o",
            color="red",
            linewidth=2,
            label="Monitoring",
        )
        ax3.set_ylabel("Accuracy", fontweight="bold")
        ax3.set_title("Accuracy Degradation", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        # 4. Alert summary
        ax4 = fig.add_subplot(gs[1, 2])
        alert_counts = [len(performance_results[s].alerts) for s in scenarios]
        colors = ["red" if c > 0 else "green" for c in alert_counts]
        ax4.bar(range(len(scenarios)), alert_counts, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(scenarios)))
        ax4.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
        ax4.set_ylabel("# Alerts", fontweight="bold")
        ax4.set_title("Performance Alerts", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

        # 5. Top drifted features (first scenario with drift)
        ax5 = fig.add_subplot(gs[2, :])

        # Find scenario with most drift
        max_drift_scenario = max(
            scenarios,
            key=lambda s: drift_results[s]["drift_summary"]["drift_percentage"],
        )

        if drift_results[max_drift_scenario]["drifted_features"]:
            from fase3.monitoring.drift_detector import (
                calculate_feature_importance_for_drift,
            )

            top_features = calculate_feature_importance_for_drift(
                drift_results[max_drift_scenario], top_n=10
            )

            features, scores = zip(*top_features)
            ax5.barh(features, scores, color="coral")
            ax5.set_xlabel("Average Drift Score", fontweight="bold")
            ax5.set_title(
                f"Top 10 Drifted Features ({max_drift_scenario})", fontweight="bold"
            )
            ax5.grid(True, alpha=0.3, axis="x")
        else:
            ax5.text(
                0.5,
                0.5,
                "No significant drift detected",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax5.set_xticks([])
            ax5.set_yticks([])

        plt.suptitle("Data Drift Monitoring Dashboard", fontweight="bold", fontsize=16)

        if save_path is None:
            save_path = self.output_dir / "drift_dashboard.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.success(f"Saved drift dashboard: {save_path}")
        plt.close()
