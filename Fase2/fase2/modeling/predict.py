"""
Model prediction and evaluation script.

This script follows Cookiecutter Data Science template and implements:
- Model evaluation on test set
- Metrics calculation
- Visualization generation
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Optional

from fase2.config import config
from fase2.plots import plot_confusion_matrix, plot_roc_curve, plot_feature_importance


def evaluate_model(
    model_path: Optional[Path] = None,
    save_results: bool = True,
    generate_plots: bool = True,
) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to saved pipeline (.pkl file)
        save_results: Whether to save predictions and metrics
        generate_plots: Whether to generate visualization plots

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    # Load pipeline
    if model_path is None:
        model_path = config.paths.models_dir / "random_forest_pipeline.pkl"

    logger.info(f"\n1️⃣ Loading pipeline from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run 'python -m fase2.modeling.train' first."
        )

    pipeline = joblib.load(model_path)
    logger.success(f"✓ Pipeline loaded: {type(pipeline).__name__}")

    # Load test data
    logger.info("\n2️⃣ Loading test data...")
    X_test_path = config.paths.processed_data_dir / "X_test.csv"
    y_test_path = config.paths.processed_data_dir / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "Test data not found. Run 'python -m fase2.features' first."
        )

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    logger.success(f"✓ Test data loaded: {X_test.shape}")

    # Make predictions
    logger.info("\n3️⃣ Making predictions...")
    y_pred = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    # Calculate metrics
    logger.info("\n4️⃣ Calculating metrics...")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))

    # Display results
    logger.success("\n✓ Evaluation complete:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if "auc_roc" in metrics:
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # Classification report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    report = classification_report(
        y_test, y_pred, target_names=["Bad Credit (0)", "Good Credit (1)"]
    )
    print(report)

    # Save results
    if save_results:
        logger.info("\n5️⃣ Saving results...")
        output_dir = config.paths.processed_data_dir

        # Save predictions
        predictions_df = pd.DataFrame({"true_label": y_test, "predicted_label": y_pred})
        if y_proba is not None:
            predictions_df["probability"] = y_proba

        pred_path = output_dir / "test_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.success(f"✓ Predictions saved to: {pred_path}")

        # Save metrics
        metrics_path = output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.success(f"✓ Metrics saved to: {metrics_path}")

    # Generate plots
    if generate_plots:
        logger.info("\n6️⃣ Generating visualizations...")

        model_name = model_path.stem.replace("_pipeline", "").replace("_", " ").title()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, model_name)

        # ROC curve
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, model_name)

        # Feature importance (if available)
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "Feature": X_test.columns,
                    "Importance": pipeline.named_steps["model"].feature_importances_,
                }
            ).sort_values("Importance", ascending=False)

            plot_feature_importance(importance_df, model_name)

        logger.success("✓ Visualizations generated")

    logger.success("\n" + "=" * 70)

    return metrics


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def predict(
        model_path: str = typer.Option(
            None, "--model-path", "-m", help="Path to trained pipeline .pkl file"
        ),
        no_plots: bool = typer.Option(
            False, "--no-plots", help="Skip generating visualization plots"
        ),
    ):
        """
        Evaluate trained model on test set.

        Example:
            python -m fase2.modeling.predict
            python -m fase2.modeling.predict --model-path models/logistic_regression_pipeline.pkl
            python -m fase2.modeling.predict --no-plots
        """
        model_path_obj = Path(model_path) if model_path else None
        evaluate_model(model_path=model_path_obj, generate_plots=not no_plots)

    app()


if __name__ == "__main__":
    main()
