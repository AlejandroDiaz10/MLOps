"""
Model inference and prediction utilities
"""

from pathlib import Path

from loguru import logger
import typer
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import json

from fase2.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# ============= FUNCIONES REUTILIZABLES =============


def load_model(model_path: Path):
    """Load trained model from disk"""
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.success(f"✓ Model loaded: {type(model).__name__}")
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions with loaded model"""
    logger.info(f"Making predictions on {len(X)} samples...")
    predictions = model.predict(X)
    logger.success(f"✓ Predictions complete")
    return predictions


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Get prediction probabilities"""
    if hasattr(model, "predict_proba"):
        logger.info(f"Computing prediction probabilities...")
        probabilities = model.predict_proba(X)[:, 1]
        logger.success(f"✓ Probabilities computed")
        return probabilities
    else:
        logger.warning(f"⚠ Model does not support predict_proba")
        return None


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Comprehensive model evaluation
    Returns dictionary of metrics
    """
    logger.info("Evaluating model performance...")

    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["auc_roc"] = None

    logger.success("✓ Model evaluation complete")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if metrics["auc_roc"]:
        logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    return metrics


# ============= CLI COMMAND =============


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
    model_path: Path = MODELS_DIR / "random_forest.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Perform inference using a trained model.

    Loads test data, makes predictions, and evaluates performance.
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL INFERENCE")
    logger.info("=" * 60)

    # Load test data
    logger.info("Loading test data...")
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(labels_path).values.ravel()
    logger.success(f"✓ Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")

    # Load model
    model = load_model(model_path)

    # Make predictions
    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save predictions
    predictions_df = pd.DataFrame({"true_label": y_test, "predicted_label": y_pred})

    if y_proba is not None:
        predictions_df["probability"] = y_proba

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"✓ Predictions saved to: {predictions_path}")

    # Save metrics
    metrics_path = predictions_path.parent / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"✓ Metrics saved to: {metrics_path}")

    # Print classification report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=["Bad Credit (0)", "Good Credit (1)"]
        )
    )

    logger.success("=" * 60)
    logger.success("✓ MODEL INFERENCE COMPLETE")
    logger.success("=" * 60)


if __name__ == "__main__":
    app()
