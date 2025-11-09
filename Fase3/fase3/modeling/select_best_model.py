"""
Select best model based on test AUC-ROC score.

This script:
1. Reads metrics from all trained models
2. Selects the model with highest test AUC-ROC
3. Copies it to best_model_pipeline.pkl
4. Registers it in MLflow as the production model
"""

import json
import shutil
from pathlib import Path
from loguru import logger
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

from fase3.config import config


def select_best_model() -> None:
    """
    Select best model based on test AUC-ROC and copy to best_model_pipeline.pkl
    """
    logger.info("=" * 70)
    logger.info("SELECTING BEST MODEL")
    logger.info("=" * 70)

    metrics_dir = config.paths.proj_root / "reports" / "metrics"
    models_dir = config.paths.models_dir

    # Model names to compare
    model_names = ["random_forest", "logistic_regression", "decision_tree"]

    # Load metrics for all models
    models_metrics = {}

    for model_name in model_names:
        metrics_file = metrics_dir / f"{model_name}_metrics.json"

        if not metrics_file.exists():
            logger.warning(f"⚠️ Metrics not found for {model_name}, skipping...")
            continue

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        models_metrics[model_name] = metrics
        logger.info(f"  {model_name}: AUC-ROC = {metrics.get('test_auc_roc', 0):.4f}")

    if not models_metrics:
        raise ValueError("No model metrics found. Train models first.")

    # Select best model based on test AUC-ROC
    best_model_name = max(
        models_metrics.keys(), key=lambda x: models_metrics[x].get("test_auc_roc", 0)
    )
    best_metrics = models_metrics[best_model_name]

    logger.success(f"\n🏆 Best Model: {best_model_name.upper()}")
    logger.info(f"   Test AUC-ROC: {best_metrics['test_auc_roc']:.4f}")
    logger.info(f"   Test Accuracy: {best_metrics['test_accuracy']:.4f}")
    logger.info(f"   Test Precision: {best_metrics['test_precision']:.4f}")
    logger.info(f"   Test Recall: {best_metrics['test_recall']:.4f}")
    logger.info(f"   Test F1-Score: {best_metrics['test_f1_score']:.4f}")

    # Copy best model to best_model_pipeline.pkl
    best_model_path = models_dir / f"{best_model_name}_pipeline.pkl"
    best_model_dest = models_dir / "best_model_pipeline.pkl"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")

    shutil.copy(best_model_path, best_model_dest)
    logger.success(f"✅ Copied {best_model_name} to: {best_model_dest}")

    # Copy metadata
    best_metadata_path = models_dir / f"{best_model_name}_pipeline_metadata.json"
    best_metadata_dest = models_dir / "best_model_pipeline_metadata.json"

    if best_metadata_path.exists():
        with open(best_metadata_path, "r") as f:
            metadata = json.load(f)

        # Add selection info
        metadata["selection_info"] = {
            "selected_from": model_names,
            "selection_date": datetime.now().isoformat(),
            "selection_criterion": "test_auc_roc",
            "test_auc_roc": best_metrics["test_auc_roc"],
        }

        with open(best_metadata_dest, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"✅ Updated metadata: {best_metadata_dest}")

    # Save selection summary for DVC
    selection_summary = {
        "best_model_name": best_model_name,
        "selection_date": datetime.now().isoformat(),
        "selection_criterion": "test_auc_roc",
        "models_compared": list(models_metrics.keys()),
        "best_metrics": best_metrics,
        "all_metrics": {
            name: {
                "test_auc_roc": metrics.get("test_auc_roc", 0),
                "test_accuracy": metrics.get("test_accuracy", 0),
            }
            for name, metrics in models_metrics.items()
        },
    }

    selection_file = metrics_dir / "best_model_selection.json"
    with open(selection_file, "w") as f:
        json.dump(selection_summary, f, indent=2)

    logger.success(f"✅ Selection summary saved: {selection_file}")

    # Register in MLflow as Production model
    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        logger.info("\n📊 Registering in MLflow as Production model...")

        # Load the best model
        best_pipeline = joblib.load(best_model_dest)

        # Register with a special name
        production_model_name = f"{config.mlflow.experiment_name}_production"

        with mlflow.start_run(
            run_name=f"best_model_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log all comparison metrics
            for model_name, metrics in models_metrics.items():
                mlflow.log_metric(f"{model_name}_test_auc_roc", metrics.get("test_auc_roc", 0))
                mlflow.log_metric(f"{model_name}_test_accuracy", metrics.get("test_accuracy", 0))

            # Log selection decision
            mlflow.log_param("best_model_selected", best_model_name)
            mlflow.log_param("selection_criterion", "test_auc_roc")
            mlflow.log_metric("best_test_auc_roc", best_metrics["test_auc_roc"])

            # Log the production model
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path="model",
                registered_model_name=production_model_name,
            )

            logger.success(f"✅ Registered as: {production_model_name}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to register in MLflow: {e}")

    logger.info("\n" + "🎉" * 35)
    logger.success("🎉 BEST MODEL SELECTION COMPLETED!")
    logger.info("🎉" * 35)


if __name__ == "__main__":
    select_best_model()
