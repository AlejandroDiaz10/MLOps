"""
Data Drift Simulator - Generate synthetic drifted data.

This module creates monitoring datasets with altered distributions to simulate
real-world data drift scenarios:
- Feature drift: Changes in feature distributions
- Concept drift: Changes in target relationship
- Missing features: Simulated data quality issues
- Seasonal drift: Time-based distribution changes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):
    """Types of data drift to simulate."""

    MEAN_SHIFT = "mean_shift"  # Shift in feature means
    VARIANCE_CHANGE = "variance_change"  # Change in feature variance
    MISSING_FEATURES = "missing_features"  # Introduce missing values
    FEATURE_CORRELATION = "feature_correlation"  # Change correlations
    OUTLIERS = "outliers"  # Introduce outliers
    CATEGORICAL_SHIFT = "categorical_shift"  # Change categorical distributions
    CONCEPT_DRIFT = "concept_drift"  # Change X-y relationship


@dataclass
class DriftConfig:
    """Configuration for drift simulation."""

    drift_type: DriftType
    severity: float = 0.3  # 0.0 (no drift) to 1.0 (severe drift)
    affected_features: Optional[List[str]] = None  # None = all features
    seed: int = 42


class DataDriftSimulator:
    """
    Simulate various types of data drift for monitoring.

    Example usage:
        simulator = DataDriftSimulator(reference_data)
        drifted_data = simulator.generate_drift(
            DriftConfig(DriftType.MEAN_SHIFT, severity=0.5)
        )
    """

    def __init__(self, reference_data: pd.DataFrame, target_col: str = "credit_risk"):
        """
        Initialize simulator with reference (training) data.

        Args:
            reference_data: Original training/validation data
            target_col: Name of target column
        """
        self.reference_data = reference_data.copy()
        self.target_col = target_col

        # Separate features and target
        if target_col in reference_data.columns:
            self.X_ref = reference_data.drop(columns=[target_col])
            self.y_ref = reference_data[target_col]
        else:
            self.X_ref = reference_data
            self.y_ref = None

        # Compute reference statistics
        self._compute_reference_stats()

        logger.info(
            f"DataDriftSimulator initialized with {len(reference_data)} reference samples"
        )

    def _compute_reference_stats(self):
        """Compute statistics from reference data."""
        self.ref_means = self.X_ref.mean()
        self.ref_stds = self.X_ref.std()
        self.ref_mins = self.X_ref.min()
        self.ref_maxs = self.X_ref.max()

        # Identify numeric and categorical features
        self.numeric_features = self.X_ref.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_features = self.X_ref.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

    def generate_drift(
        self, config: DriftConfig, n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate drifted data based on configuration.

        Args:
            config: Drift configuration
            n_samples: Number of samples to generate (default: same as reference)

        Returns:
            DataFrame with drifted data
        """
        if n_samples is None:
            n_samples = len(self.reference_data)

        np.random.seed(config.seed)

        # Start with a sample from reference data (with replacement)
        indices = np.random.choice(len(self.X_ref), size=n_samples, replace=True)
        X_drift = self.X_ref.iloc[indices].copy().reset_index(drop=True)

        if self.y_ref is not None:
            y_drift = self.y_ref.iloc[indices].copy().reset_index(drop=True)
        else:
            y_drift = None

        # Determine affected features
        if config.affected_features is None:
            affected_features = self.numeric_features
        else:
            affected_features = [
                f for f in config.affected_features if f in self.numeric_features
            ]

        # Apply drift based on type
        if config.drift_type == DriftType.MEAN_SHIFT:
            X_drift = self._apply_mean_shift(
                X_drift, affected_features, config.severity
            )

        elif config.drift_type == DriftType.VARIANCE_CHANGE:
            X_drift = self._apply_variance_change(
                X_drift, affected_features, config.severity
            )

        elif config.drift_type == DriftType.MISSING_FEATURES:
            X_drift = self._apply_missing_features(
                X_drift, affected_features, config.severity
            )

        elif config.drift_type == DriftType.OUTLIERS:
            X_drift = self._apply_outliers(X_drift, affected_features, config.severity)

        elif config.drift_type == DriftType.CATEGORICAL_SHIFT:
            X_drift = self._apply_categorical_shift(X_drift, config.severity)

        elif config.drift_type == DriftType.CONCEPT_DRIFT:
            X_drift, y_drift = self._apply_concept_drift(
                X_drift, y_drift, config.severity
            )

        # Combine features and target
        if y_drift is not None:
            result = X_drift.copy()
            result[self.target_col] = y_drift
        else:
            result = X_drift

        logger.success(
            f"Generated {len(result)} samples with {config.drift_type.value} drift (severity={config.severity})"
        )

        return result

    def _apply_mean_shift(
        self, X: pd.DataFrame, features: List[str], severity: float
    ) -> pd.DataFrame:
        """Apply mean shift to features."""
        X_shifted = X.copy()

        for feature in features:
            # Shift mean by severity * std
            shift_amount = severity * self.ref_stds[feature]
            X_shifted[feature] = X_shifted[feature] + shift_amount

            logger.debug(f"  {feature}: mean shift = +{shift_amount:.2f}")

        return X_shifted

    def _apply_variance_change(
        self, X: pd.DataFrame, features: List[str], severity: float
    ) -> pd.DataFrame:
        """Apply variance change to features."""
        X_changed = X.copy()

        for feature in features:
            # Increase variance by (1 + severity)
            scale_factor = 1 + severity
            mean = X_changed[feature].mean()
            X_changed[feature] = mean + (X_changed[feature] - mean) * scale_factor

            logger.debug(f"  {feature}: variance scaled by {scale_factor:.2f}")

        return X_changed

    def _apply_missing_features(
        self, X: pd.DataFrame, features: List[str], severity: float
    ) -> pd.DataFrame:
        """Introduce missing values in features."""
        X_missing = X.copy()

        for feature in features:
            # Introduce missing values with probability = severity
            n_missing = int(len(X_missing) * severity)
            missing_indices = np.random.choice(
                len(X_missing), size=n_missing, replace=False
            )
            X_missing.loc[missing_indices, feature] = np.nan

            logger.debug(
                f"  {feature}: {n_missing} missing values ({severity*100:.1f}%)"
            )

        return X_missing

    def _apply_outliers(
        self, X: pd.DataFrame, features: List[str], severity: float
    ) -> pd.DataFrame:
        """Introduce outliers in features."""
        X_outliers = X.copy()

        for feature in features:
            # Introduce outliers with probability = severity
            n_outliers = int(len(X_outliers) * severity)
            outlier_indices = np.random.choice(
                len(X_outliers), size=n_outliers, replace=False
            )

            # Outliers are 3-5 standard deviations away
            outlier_values = (
                self.ref_means[feature]
                + np.random.choice([-1, 1], size=n_outliers)
                * np.random.uniform(3, 5, size=n_outliers)
                * self.ref_stds[feature]
            )

            X_outliers.loc[outlier_indices, feature] = outlier_values

            logger.debug(f"  {feature}: {n_outliers} outliers ({severity*100:.1f}%)")

        return X_outliers

    def _apply_categorical_shift(
        self, X: pd.DataFrame, severity: float
    ) -> pd.DataFrame:
        """Apply shift to categorical feature distributions."""
        X_shifted = X.copy()

        # For German Credit, categorical features are encoded as integers
        # Shift them towards extreme values
        for feature in self.numeric_features:
            if X_shifted[feature].nunique() < 10:  # Likely categorical
                # Shift towards max value with probability = severity
                shift_mask = np.random.random(len(X_shifted)) < severity
                max_val = self.ref_maxs[feature]
                X_shifted.loc[shift_mask, feature] = max_val

                n_shifted = shift_mask.sum()
                logger.debug(f"  {feature}: {n_shifted} values shifted to {max_val}")

        return X_shifted

    def _apply_concept_drift(
        self, X: pd.DataFrame, y: Optional[pd.Series], severity: float
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Apply concept drift (change X-y relationship)."""
        if y is None:
            logger.warning("Cannot apply concept drift without target variable")
            return X, y

        y_drifted = y.copy()

        # Flip labels with probability = severity
        n_flips = int(len(y_drifted) * severity)
        flip_indices = np.random.choice(len(y_drifted), size=n_flips, replace=False)
        y_drifted.iloc[flip_indices] = 1 - y_drifted.iloc[flip_indices]

        logger.debug(f"  Flipped {n_flips} labels ({severity*100:.1f}%)")

        return X, y_drifted

    def generate_multiple_drifts(
        self, scenarios: List[Dict], n_samples_per_scenario: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple drift scenarios for comprehensive testing.

        Args:
            scenarios: List of scenario configs (drift_type, severity, name)
            n_samples_per_scenario: Samples per scenario

        Returns:
            Dictionary mapping scenario names to drifted datasets
        """
        results = {}

        for scenario in scenarios:
            config = DriftConfig(
                drift_type=DriftType(scenario["drift_type"]),
                severity=scenario["severity"],
                seed=scenario.get("seed", 42),
            )

            name = scenario.get("name", f"{config.drift_type.value}_{config.severity}")
            results[name] = self.generate_drift(config, n_samples_per_scenario)

        logger.success(f"Generated {len(results)} drift scenarios")
        return results


def create_monitoring_scenarios(
    reference_data: pd.DataFrame, target_col: str = "credit_risk"
) -> Dict[str, pd.DataFrame]:
    """
    Create a comprehensive set of monitoring scenarios.

    Args:
        reference_data: Original validation/test data
        target_col: Target column name

    Returns:
        Dictionary with scenario name -> drifted data
    """
    simulator = DataDriftSimulator(reference_data, target_col)

    scenarios = [
        # No drift (baseline)
        {"name": "baseline", "drift_type": "mean_shift", "severity": 0.0},
        # Mild drift
        {"name": "mild_mean_shift", "drift_type": "mean_shift", "severity": 0.2},
        {
            "name": "mild_variance_change",
            "drift_type": "variance_change",
            "severity": 0.2,
        },
        # Moderate drift
        {"name": "moderate_mean_shift", "drift_type": "mean_shift", "severity": 0.5},
        {"name": "moderate_outliers", "drift_type": "outliers", "severity": 0.1},
        # Severe drift
        {"name": "severe_mean_shift", "drift_type": "mean_shift", "severity": 0.8},
        {"name": "severe_missing", "drift_type": "missing_features", "severity": 0.3},
        # Concept drift
        {"name": "mild_concept_drift", "drift_type": "concept_drift", "severity": 0.1},
        {
            "name": "severe_concept_drift",
            "drift_type": "concept_drift",
            "severity": 0.3,
        },
    ]

    return simulator.generate_multiple_drifts(scenarios, n_samples_per_scenario=200)
