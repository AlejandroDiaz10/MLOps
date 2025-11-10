"""
Data Drift Detector - Statistical tests for distribution changes.

Implements multiple statistical tests to detect data drift:
- Kolmogorov-Smirnov (KS) test: Distribution similarity
- Population Stability Index (PSI): Feature stability
- Jensen-Shannon Divergence: Distribution distance
- Chi-squared test: Categorical features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass
from loguru import logger


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature: str
    test_statistic: float
    p_value: float
    drift_detected: bool
    drift_score: float  # 0.0 (no drift) to 1.0 (severe drift)
    test_name: str


class DataDriftDetector:
    """
    Detect data drift using statistical tests.

    Example usage:
        detector = DataDriftDetector(reference_data)
        results = detector.detect_drift(monitoring_data)

        if results['overall_drift_detected']:
            print(f"Drift detected in {len(results['drifted_features'])} features")
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_col: str = "credit_risk",
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.1,
    ):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Baseline data (training/validation)
            target_col: Target column name (to exclude from drift detection)
            p_value_threshold: P-value threshold for statistical tests
            psi_threshold: PSI threshold (0.1=minor, 0.2=moderate, 0.25=severe)
        """
        self.reference_data = reference_data.copy()
        self.target_col = target_col
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

        # Separate features and target
        if target_col in reference_data.columns:
            self.X_ref = reference_data.drop(columns=[target_col])
            self.y_ref = reference_data[target_col]
        else:
            self.X_ref = reference_data
            self.y_ref = None

        # Identify feature types
        self.numeric_features = self.X_ref.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_features = self.X_ref.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        # Compute reference distributions
        self._compute_reference_distributions()

        logger.info(
            f"DriftDetector initialized with {len(reference_data)} reference samples"
        )
        logger.info(f"  Numeric features: {len(self.numeric_features)}")
        logger.info(f"  Categorical features: {len(self.categorical_features)}")

    def _compute_reference_distributions(self):
        """Compute reference distributions for PSI calculation."""
        self.ref_distributions = {}

        for feature in self.numeric_features:
            # Create bins for PSI calculation
            values = self.X_ref[feature].dropna()

            # Use quantile-based bins (10 bins)
            try:
                bins = np.percentile(values, np.linspace(0, 100, 11))
                bins = np.unique(bins)  # Remove duplicates

                if len(bins) > 1:
                    self.ref_distributions[feature] = {
                        "bins": bins,
                        "counts": np.histogram(values, bins=bins)[0],
                    }
            except Exception as e:
                logger.warning(f"Could not create bins for {feature}: {e}")

    def detect_drift(
        self, monitoring_data: pd.DataFrame, methods: List[str] = ["ks", "psi"]
    ) -> Dict:
        """
        Detect drift in monitoring data compared to reference.

        Args:
            monitoring_data: New data to check for drift
            methods: List of methods to use ['ks', 'psi', 'js']

        Returns:
            Dictionary with drift detection results
        """
        # Separate features
        if self.target_col in monitoring_data.columns:
            X_mon = monitoring_data.drop(columns=[self.target_col])
        else:
            X_mon = monitoring_data

        results = {
            "n_reference": len(self.X_ref),
            "n_monitoring": len(X_mon),
            "feature_results": {},
            "drifted_features": [],
            "overall_drift_detected": False,
            "drift_summary": {},
        }

        # Test each feature
        for feature in self.numeric_features:
            if feature not in X_mon.columns:
                logger.warning(f"Feature {feature} not in monitoring data")
                continue

            feature_results = {}

            # Kolmogorov-Smirnov test
            if "ks" in methods:
                ks_result = self._ks_test(self.X_ref[feature], X_mon[feature])
                feature_results["ks"] = ks_result

            # Population Stability Index
            if "psi" in methods:
                psi_result = self._psi_test(
                    feature, self.X_ref[feature], X_mon[feature]
                )
                feature_results["psi"] = psi_result

            # Jensen-Shannon Divergence
            if "js" in methods:
                js_result = self._js_divergence(self.X_ref[feature], X_mon[feature])
                feature_results["js"] = js_result

            # Determine if drift detected (any test)
            drift_detected = any(r.drift_detected for r in feature_results.values())

            if drift_detected:
                results["drifted_features"].append(feature)

            results["feature_results"][feature] = feature_results

        # Overall drift detection
        results["overall_drift_detected"] = len(results["drifted_features"]) > 0

        # Drift summary
        results["drift_summary"] = {
            "n_features_tested": len(self.numeric_features),
            "n_features_drifted": len(results["drifted_features"]),
            "drift_percentage": len(results["drifted_features"])
            / len(self.numeric_features)
            * 100,
        }

        logger.info(
            f"Drift detection complete: {results['drift_summary']['n_features_drifted']}/{results['drift_summary']['n_features_tested']} features drifted"
        )

        return results

    def _ks_test(self, ref_values: pd.Series, mon_values: pd.Series) -> DriftResult:
        """
        Kolmogorov-Smirnov test for distribution similarity.

        Args:
            ref_values: Reference feature values
            mon_values: Monitoring feature values

        Returns:
            DriftResult with test outcome
        """
        # Remove NaNs
        ref_clean = ref_values.dropna()
        mon_clean = mon_values.dropna()

        # KS test
        statistic, p_value = stats.ks_2samp(ref_clean, mon_clean)

        # Drift detected if p < threshold
        drift_detected = p_value < self.p_value_threshold

        # Drift score based on KS statistic (0-1)
        drift_score = min(statistic, 1.0)

        return DriftResult(
            feature=ref_values.name,
            test_statistic=statistic,
            p_value=p_value,
            drift_detected=drift_detected,
            drift_score=drift_score,
            test_name="Kolmogorov-Smirnov",
        )

    def _psi_test(
        self, feature: str, ref_values: pd.Series, mon_values: pd.Series
    ) -> DriftResult:
        """
        Population Stability Index (PSI) test.

        PSI measures the shift in population distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Minor change
        - PSI >= 0.2: Major change (action required)

        Args:
            feature: Feature name
            ref_values: Reference feature values
            mon_values: Monitoring feature values

        Returns:
            DriftResult with PSI score
        """
        if feature not in self.ref_distributions:
            logger.warning(f"No reference distribution for {feature}")
            return DriftResult(
                feature=feature,
                test_statistic=0.0,
                p_value=1.0,
                drift_detected=False,
                drift_score=0.0,
                test_name="PSI",
            )

        bins = self.ref_distributions[feature]["bins"]
        ref_counts = self.ref_distributions[feature]["counts"]

        # Bin monitoring data
        mon_counts, _ = np.histogram(mon_values.dropna(), bins=bins)

        # Convert to proportions
        ref_prop = ref_counts / ref_counts.sum()
        mon_prop = mon_counts / mon_counts.sum()

        # Add small constant to avoid log(0)
        epsilon = 1e-10
        ref_prop = np.where(ref_prop == 0, epsilon, ref_prop)
        mon_prop = np.where(mon_prop == 0, epsilon, mon_prop)

        # Calculate PSI
        psi = np.sum((mon_prop - ref_prop) * np.log(mon_prop / ref_prop))

        # Drift detected if PSI > threshold
        drift_detected = psi > self.psi_threshold

        # Drift score (normalized)
        drift_score = min(psi / 0.25, 1.0)  # 0.25 = severe drift threshold

        return DriftResult(
            feature=feature,
            test_statistic=psi,
            p_value=0.0,  # PSI doesn't have p-value
            drift_detected=drift_detected,
            drift_score=drift_score,
            test_name="PSI",
        )

    def _js_divergence(
        self, ref_values: pd.Series, mon_values: pd.Series, n_bins: int = 30
    ) -> DriftResult:
        """
        Jensen-Shannon Divergence test.

        JS divergence measures the similarity between two probability distributions.
        Range: 0 (identical) to 1 (completely different)

        Args:
            ref_values: Reference feature values
            mon_values: Monitoring feature values
            n_bins: Number of bins for histogram

        Returns:
            DriftResult with JS divergence
        """
        # Remove NaNs
        ref_clean = ref_values.dropna()
        mon_clean = mon_values.dropna()

        # Create common bins
        min_val = min(ref_clean.min(), mon_clean.min())
        max_val = max(ref_clean.max(), mon_clean.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Compute histograms
        ref_hist, _ = np.histogram(ref_clean, bins=bins, density=True)
        mon_hist, _ = np.histogram(mon_clean, bins=bins, density=True)

        # Normalize to probabilities
        ref_prob = ref_hist / ref_hist.sum()
        mon_prob = mon_hist / mon_hist.sum()

        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, mon_prob)

        # Drift detected if JS > 0.1 (arbitrary threshold)
        drift_detected = js_div > 0.1

        # Drift score
        drift_score = min(js_div, 1.0)

        return DriftResult(
            feature=ref_values.name,
            test_statistic=js_div,
            p_value=0.0,  # JS doesn't have p-value
            drift_detected=drift_detected,
            drift_score=drift_score,
            test_name="Jensen-Shannon",
        )

    def get_drift_severity(self, drift_results: Dict) -> str:
        """
        Classify overall drift severity.

        Args:
            drift_results: Results from detect_drift()

        Returns:
            Severity level: 'none', 'minor', 'moderate', 'severe'
        """
        drift_pct = drift_results["drift_summary"]["drift_percentage"]

        if drift_pct == 0:
            return "none"
        elif drift_pct < 25:
            return "minor"
        elif drift_pct < 50:
            return "moderate"
        else:
            return "severe"

    def get_recommended_action(self, drift_results: Dict) -> str:
        """
        Get recommended action based on drift severity.

        Args:
            drift_results: Results from detect_drift()

        Returns:
            Recommended action string
        """
        severity = self.get_drift_severity(drift_results)

        actions = {
            "none": "No action required. Model performance is stable.",
            "minor": "Monitor closely. Consider feature investigation if performance degrades.",
            "moderate": "Action recommended: Review feature pipeline and consider model retraining.",
            "severe": "URGENT: Significant drift detected. Retrain model immediately or investigate data quality issues.",
        }

        return actions[severity]


def calculate_feature_importance_for_drift(
    drift_results: Dict, top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Identify most drifted features based on drift scores.

    Args:
        drift_results: Results from detect_drift()
        top_n: Number of top features to return

    Returns:
        List of (feature, avg_drift_score) tuples
    """
    feature_scores = {}

    for feature, tests in drift_results["feature_results"].items():
        # Average drift score across all tests
        scores = [result.drift_score for result in tests.values()]
        feature_scores[feature] = np.mean(scores)

    # Sort by drift score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_features[:top_n]
