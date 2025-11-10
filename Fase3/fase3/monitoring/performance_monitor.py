"""
Performance Monitor - Track model performance degradation.

Monitors model performance on new data and compares with baseline metrics.
Detects performance degradation that may indicate data drift impact.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "n_samples": self.n_samples,
        }


@dataclass
class PerformanceComparison:
    """Comparison between baseline and monitoring metrics."""

    baseline: PerformanceMetrics
    monitoring: PerformanceMetrics
    degradation: Dict[str, float]  # Percentage change
    alerts: List[str]
    degradation_detected: bool


class PerformanceMonitor:
    """
    Monitor model performance and detect degradation.

    Example usage:
        monitor = PerformanceMonitor(model, baseline_metrics)
        result = monitor.evaluate_and_compare(X_new, y_new)

        if result.degradation_detected:
            print(f"Performance degraded: {result.alerts}")
    """

    def __init__(
        self,
        model,
        baseline_metrics: PerformanceMetrics,
        degradation_threshold: float = 0.05,  # 5% degradation triggers alert
    ):
        """
        Initialize performance monitor.

        Args:
            model: Trained model (sklearn pipeline)
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Relative degradation threshold (0.05 = 5%)
        """
        self.model = model
        self.baseline = baseline_metrics
        self.degradation_threshold = degradation_threshold

        logger.info(f"PerformanceMonitor initialized")
        logger.info(f"  Baseline AUC-ROC: {baseline_metrics.auc_roc:.4f}")
        logger.info(f"  Degradation threshold: {degradation_threshold*100:.1f}%")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> PerformanceMetrics:
        """
        Evaluate model on new data.

        Args:
            X: Features
            y: True labels

        Returns:
            PerformanceMetrics object
        """
        # Make predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = PerformanceMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1_score=f1_score(y, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y, y_proba),
            n_samples=len(y),
        )

        return metrics

    def compare_metrics(
        self, monitoring_metrics: PerformanceMetrics
    ) -> PerformanceComparison:
        """
        Compare monitoring metrics with baseline.

        Args:
            monitoring_metrics: Metrics on monitoring data

        Returns:
            PerformanceComparison object
        """
        # Calculate degradation (negative = worse)
        degradation = {
            "accuracy": (monitoring_metrics.accuracy - self.baseline.accuracy)
            / self.baseline.accuracy,
            "precision": (monitoring_metrics.precision - self.baseline.precision)
            / self.baseline.precision,
            "recall": (monitoring_metrics.recall - self.baseline.recall)
            / self.baseline.recall,
            "f1_score": (monitoring_metrics.f1_score - self.baseline.f1_score)
            / self.baseline.f1_score,
            "auc_roc": (monitoring_metrics.auc_roc - self.baseline.auc_roc)
            / self.baseline.auc_roc,
        }

        # Generate alerts
        alerts = []
        degradation_detected = False

        for metric_name, deg_value in degradation.items():
            if deg_value < -self.degradation_threshold:
                alerts.append(
                    f"{metric_name.upper()}: {deg_value*100:.2f}% degradation "
                    f"(baseline: {getattr(self.baseline, metric_name):.4f}, "
                    f"current: {getattr(monitoring_metrics, metric_name):.4f})"
                )
                degradation_detected = True

        if not degradation_detected:
            logger.success("✅ No significant performance degradation detected")
        else:
            logger.warning(
                f"⚠️  Performance degradation detected: {len(alerts)} metrics affected"
            )

        return PerformanceComparison(
            baseline=self.baseline,
            monitoring=monitoring_metrics,
            degradation=degradation,
            alerts=alerts,
            degradation_detected=degradation_detected,
        )

    def evaluate_and_compare(
        self, X: pd.DataFrame, y: pd.Series
    ) -> PerformanceComparison:
        """
        Evaluate model on new data and compare with baseline.

        Args:
            X: Features
            y: True labels

        Returns:
            PerformanceComparison object
        """
        monitoring_metrics = self.evaluate(X, y)
        return self.compare_metrics(monitoring_metrics)

    def get_severity_level(self, comparison: PerformanceComparison) -> str:
        """
        Classify degradation severity.

        Args:
            comparison: Performance comparison result

        Returns:
            Severity level: 'none', 'minor', 'moderate', 'severe'
        """
        if not comparison.degradation_detected:
            return "none"

        # Use AUC-ROC as primary metric
        auc_degradation = abs(comparison.degradation["auc_roc"])

        if auc_degradation < 0.05:
            return "minor"
        elif auc_degradation < 0.10:
            return "moderate"
        else:
            return "severe"

    def get_recommended_action(self, comparison: PerformanceComparison) -> str:
        """
        Get recommended action based on degradation severity.

        Args:
            comparison: Performance comparison result

        Returns:
            Recommended action string
        """
        severity = self.get_severity_level(comparison)

        actions = {
            "none": "No action required. Model performance is within acceptable range.",
            "minor": "Monitor closely. Investigate if degradation persists or worsens.",
            "moderate": "Action recommended: Investigate root cause (data drift, data quality). Consider model retraining.",
            "severe": "URGENT ACTION REQUIRED: Significant performance drop detected. Retrain model or roll back to previous version.",
        }

        return actions[severity]


def create_baseline_metrics(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> PerformanceMetrics:
    """
    Create baseline metrics from test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        PerformanceMetrics object
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = PerformanceMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1_score=f1_score(y_test, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_test, y_proba),
        n_samples=len(y_test),
    )

    logger.info(f"Baseline metrics created from {len(y_test)} samples")
    logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")

    return metrics


def monitor_multiple_scenarios(
    model,
    baseline_metrics: PerformanceMetrics,
    scenarios: Dict[str, Tuple[pd.DataFrame, pd.Series]],
) -> Dict[str, PerformanceComparison]:
    """
    Monitor performance across multiple drift scenarios.

    Args:
        model: Trained model
        baseline_metrics: Baseline performance
        scenarios: Dict mapping scenario name to (X, y) tuple

    Returns:
        Dict mapping scenario name to PerformanceComparison
    """
    monitor = PerformanceMonitor(model, baseline_metrics)
    results = {}

    for scenario_name, (X, y) in scenarios.items():
        logger.info(f"\n📊 Evaluating scenario: {scenario_name}")
        comparison = monitor.evaluate_and_compare(X, y)
        results[scenario_name] = comparison

        # Log summary
        severity = monitor.get_severity_level(comparison)
        logger.info(f"  Severity: {severity.upper()}")
        logger.info(
            f"  AUC-ROC: {comparison.monitoring.auc_roc:.4f} (Δ {comparison.degradation['auc_roc']*100:+.2f}%)"
        )

    return results
