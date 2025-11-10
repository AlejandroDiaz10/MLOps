"""
Main Drift Monitoring Script.

Orchestrates complete drift monitoring workflow:
1. Load reference data and model
2. Generate drift scenarios
3. Detect drift
4. Monitor performance
5. Generate reports and visualizations
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict
import typer
from loguru import logger

# IMPORTS CORRECTOS - Usar imports absolutos desde fase3.monitoring
from fase3.monitoring.drift_simulator import (
    DataDriftSimulator,
    DriftType,
    DriftConfig,
    create_monitoring_scenarios,
)
from fase3.monitoring.drift_detector import (
    DataDriftDetector,
    calculate_feature_importance_for_drift,
)
from fase3.monitoring.performance_monitor import (
    PerformanceMonitor,
    create_baseline_metrics,
    monitor_multiple_scenarios,
)
from fase3.monitoring.drift_visualizer import DriftVisualizer
from fase3.config import config

app = typer.Typer(help="Data Drift Monitoring System")


def load_data_and_model():
    """Load reference data and trained model."""
    logger.info("Loading reference data and model...")

    # Build paths manually (config doesn't have X_test_path)
    processed_dir = config.paths.processed_data_dir
    X_test_path = processed_dir / "X_test.csv"
    y_test_path = processed_dir / "y_test.csv"

    # Check if files exist
    if not X_test_path.exists():
        logger.error(f"❌ X_test.csv not found at: {X_test_path}")
        logger.info("   Run: python -m fase3.features")
        raise FileNotFoundError(f"X_test.csv not found at {X_test_path}")

    if not y_test_path.exists():
        logger.error(f"❌ y_test.csv not found at: {y_test_path}")
        logger.info("   Run: python -m fase3.features")
        raise FileNotFoundError(f"y_test.csv not found at {y_test_path}")

    # Load test data as reference
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Load best model
    model_path = config.paths.models_dir / "best_model_pipeline.pkl"

    if not model_path.exists():
        logger.error(f"❌ Model not found at: {model_path}")
        logger.info("   Run: python -m fase3.modeling.select_best_model")
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)

    # Combine for reference
    reference_data = X_test.copy()
    reference_data["credit_risk"] = y_test

    logger.success(f"Loaded {len(reference_data)} reference samples")
    logger.success(f"Loaded model: {model_path}")

    return reference_data, X_test, y_test, model


def generate_drift_scenarios(reference_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive drift scenarios."""
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING DRIFT SCENARIOS")
    logger.info("=" * 70)

    scenarios = create_monitoring_scenarios(reference_data)

    logger.success(f"Generated {len(scenarios)} drift scenarios")
    for name in scenarios.keys():
        logger.info(f"  - {name}: {len(scenarios[name])} samples")

    return scenarios


def detect_drift_in_scenarios(
    reference_data: pd.DataFrame, scenarios: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """Detect drift in all scenarios."""
    logger.info("\n" + "=" * 70)
    logger.info("DETECTING DATA DRIFT")
    logger.info("=" * 70)

    # Initialize detector
    detector = DataDriftDetector(
        reference_data,
        target_col="credit_risk",
        p_value_threshold=0.05,
        psi_threshold=0.1,
    )

    # Detect drift in each scenario
    drift_results = {}

    for scenario_name, scenario_data in scenarios.items():
        logger.info(f"\n📊 Analyzing scenario: {scenario_name}")

        results = detector.detect_drift(scenario_data, methods=["ks", "psi"])
        drift_results[scenario_name] = results

        # Log summary
        severity = detector.get_drift_severity(results)
        action = detector.get_recommended_action(results)

        logger.info(f"  Drift detected: {results['overall_drift_detected']}")
        logger.info(
            f"  Features drifted: {len(results['drifted_features'])}/{results['drift_summary']['n_features_tested']}"
        )
        logger.info(f"  Severity: {severity.upper()}")
        logger.info(f"  Action: {action[:80]}...")

        # Top drifted features
        if results["drifted_features"]:
            top_features = calculate_feature_importance_for_drift(results, top_n=3)
            logger.info(f"  Top drifted features:")
            for feature, score in top_features:
                logger.info(f"    - {feature}: {score:.3f}")

    return drift_results


def monitor_performance_in_scenarios(
    model,
    baseline_X: pd.DataFrame,
    baseline_y: pd.Series,
    scenarios: Dict[str, pd.DataFrame],
) -> Dict:
    """Monitor model performance across scenarios."""
    logger.info("\n" + "=" * 70)
    logger.info("MONITORING MODEL PERFORMANCE")
    logger.info("=" * 70)

    # Create baseline metrics
    baseline_metrics = create_baseline_metrics(model, baseline_X, baseline_y)

    # Prepare scenarios with labels
    scenarios_with_labels = {}
    for name, data in scenarios.items():
        X = data.drop(columns=["credit_risk"])
        y = data["credit_risk"]
        scenarios_with_labels[name] = (X, y)

    # Monitor performance
    performance_results = monitor_multiple_scenarios(
        model, baseline_metrics, scenarios_with_labels
    )

    return performance_results


def generate_reports_and_visualizations(
    drift_results: Dict,
    performance_results: Dict,
    scenarios: Dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
):
    """Generate comprehensive reports and visualizations."""
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING REPORTS AND VISUALIZATIONS")
    logger.info("=" * 70)

    # Create visualizer
    viz = DriftVisualizer(output_dir="reports/figures/drift")

    # 1. Feature distributions (compare baseline vs severe drift)
    logger.info("Creating feature distribution plots...")
    severe_scenario = "severe_mean_shift"
    if severe_scenario in scenarios:
        X_ref = reference_data.drop(columns=["credit_risk"])
        X_mon = scenarios[severe_scenario].drop(columns=["credit_risk"])

        # Get top drifted features
        top_features = calculate_feature_importance_for_drift(
            drift_results[severe_scenario], top_n=6
        )
        feature_names = [f[0] for f in top_features]

        viz.plot_feature_distributions(
            X_ref,
            X_mon,
            features=feature_names,
            save_path="reports/figures/drift/feature_distributions_severe.png",
        )

    # 2. Drift heatmap (for a representative scenario)
    logger.info("Creating drift heatmap...")
    moderate_scenario = "moderate_mean_shift"
    if moderate_scenario in drift_results:
        viz.plot_drift_heatmap(
            drift_results[moderate_scenario],
            save_path="reports/figures/drift/drift_heatmap_moderate.png",
        )

    # 3. Performance comparison
    logger.info("Creating performance comparison plot...")
    viz.plot_performance_comparison(
        performance_results,
        save_path="reports/figures/drift/performance_comparison.png",
    )

    # 4. Dashboard
    logger.info("Creating drift monitoring dashboard...")
    viz.plot_drift_summary_dashboard(
        drift_results,
        performance_results,
        save_path="reports/figures/drift/drift_dashboard.png",
    )

    # 5. JSON Report
    logger.info("Generating JSON report...")
    report = create_json_report(drift_results, performance_results)

    report_path = Path("reports/drift_monitoring_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.success(f"Saved JSON report: {report_path}")

    logger.success("\n✅ All reports and visualizations generated!")
    logger.info("\n📁 Outputs:")
    logger.info("  - reports/figures/drift/feature_distributions_severe.png")
    logger.info("  - reports/figures/drift/drift_heatmap_moderate.png")
    logger.info("  - reports/figures/drift/performance_comparison.png")
    logger.info("  - reports/figures/drift/drift_dashboard.png")
    logger.info("  - reports/drift_monitoring_report.json")


def create_json_report(drift_results: Dict, performance_results: Dict) -> Dict:
    """Create comprehensive JSON report."""
    report = {
        "summary": {
            "n_scenarios": len(drift_results),
            "scenarios_with_drift": sum(
                1 for r in drift_results.values() if r["overall_drift_detected"]
            ),
            "scenarios_with_degradation": sum(
                1 for r in performance_results.values() if r.degradation_detected
            ),
        },
        "scenarios": {},
    }

    for scenario_name in drift_results.keys():
        drift_res = drift_results[scenario_name]
        perf_res = performance_results[scenario_name]

        report["scenarios"][scenario_name] = {
            "drift": {
                "detected": drift_res["overall_drift_detected"],
                "n_features_drifted": len(drift_res["drifted_features"]),
                "drift_percentage": drift_res["drift_summary"]["drift_percentage"],
                "drifted_features": drift_res["drifted_features"],
            },
            "performance": {
                "baseline": perf_res.baseline.to_dict(),
                "monitoring": perf_res.monitoring.to_dict(),
                "degradation": {k: float(v) for k, v in perf_res.degradation.items()},
                "degradation_detected": perf_res.degradation_detected,
                "n_alerts": len(perf_res.alerts),
                "alerts": perf_res.alerts,
            },
        }

    return report


@app.command()
def run(
    generate_scenarios: bool = typer.Option(True, help="Generate drift scenarios"),
    detect_drift: bool = typer.Option(True, help="Detect drift"),
    monitor_performance: bool = typer.Option(True, help="Monitor performance"),
    create_visualizations: bool = typer.Option(True, help="Create visualizations"),
):
    """
    Run complete drift monitoring workflow.

    Example:
        python -m fase3.monitoring.monitor_drift run
    """
    logger.info("=" * 70)
    logger.info("DATA DRIFT MONITORING SYSTEM")
    logger.info("=" * 70)

    # Load data and model
    try:
        reference_data, X_test, y_test, model = load_data_and_model()
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        logger.info("\n📝 Prerequisites:")
        logger.info("   1. Run data preparation: python -m fase3.dataset")
        logger.info("   2. Run feature engineering: python -m fase3.features")
        logger.info("   3. Train models: dvc repro")
        logger.info(
            "   4. Select best model: python -m fase3.modeling.select_best_model"
        )
        return

    scenarios = None
    drift_results = None
    performance_results = None

    # Generate scenarios
    if generate_scenarios:
        scenarios = generate_drift_scenarios(reference_data)

    # Detect drift
    if detect_drift and scenarios:
        drift_results = detect_drift_in_scenarios(reference_data, scenarios)

    # Monitor performance
    if monitor_performance and scenarios:
        performance_results = monitor_performance_in_scenarios(
            model, X_test, y_test, scenarios
        )

    # Create visualizations
    if create_visualizations and drift_results and performance_results:
        generate_reports_and_visualizations(
            drift_results, performance_results, scenarios, reference_data
        )

    logger.info("\n" + "=" * 70)
    logger.success("✅ DRIFT MONITORING COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
