"""
Model evaluation and prediction class
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
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
import joblib
import json

from fase2.config import config
from fase2.exceptions import ModelNotFoundError


class ModelEvaluator:
    """
    Handles model evaluation and prediction operations.

    Attributes:
        config: Configuration object
        model: Loaded model instance
        X_test, y_test: Test data
        y_pred: Predictions
        y_proba: Prediction probabilities
        metrics: Evaluation metrics dictionary
    """

    def __init__(self, config_obj=None):
        """Initialize ModelEvaluator"""
        self.config = config_obj or config
        self.model = None
        self.model_path: Optional[Path] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_proba: Optional[np.ndarray] = None
        self.metrics: Optional[Dict] = None
        logger.debug("ModelEvaluator initialized")

    def load_model(self, model_path: Optional[Path] = None) -> "ModelEvaluator":
        """
        Load trained model from disk

        Args:
            model_path: Path to model file

        Returns:
            self: For method chaining
        """
        if model_path is None:
            model_path = self.config.paths.models_dir / "random_forest.pkl"

        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        self.model_path = model_path

        logger.success(f"✓ Model loaded: {type(self.model).__name__}")

        return self

    def load_test_data(
        self, X_test_path: Optional[Path] = None, y_test_path: Optional[Path] = None
    ) -> "ModelEvaluator":
        """
        Load test data from CSV files

        Args:
            X_test_path: Path to test features
            y_test_path: Path to test labels

        Returns:
            self: For method chaining
        """
        if X_test_path is None:
            X_test_path = self.config.paths.processed_data_dir / "X_test.csv"
        if y_test_path is None:
            y_test_path = self.config.paths.processed_data_dir / "y_test.csv"

        logger.info("Loading test data...")

        self.X_test = pd.read_csv(X_test_path)
        self.y_test = pd.read_csv(y_test_path).values.ravel()

        logger.success(
            f"✓ Test data loaded: X_test {self.X_test.shape}, y_test {self.y_test.shape}"
        )

        return self

    def predict(self) -> "ModelEvaluator":
        """
        Make predictions on test data

        Returns:
            self: For method chaining
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        if self.X_test is None:
            raise ValueError("No test data loaded. Call load_test_data() first.")

        logger.info(f"Making predictions on {len(self.X_test)} samples...")

        self.y_pred = self.model.predict(self.X_test)

        # Get probabilities if model supports it
        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
            logger.success("✓ Predictions and probabilities computed")
        else:
            logger.success("✓ Predictions computed (no probabilities available)")

        return self

    def evaluate(self) -> "ModelEvaluator":
        """
        Evaluate model performance

        Returns:
            self: For method chaining
        """
        if self.y_pred is None:
            raise ValueError("No predictions. Call predict() first.")

        logger.info("Evaluating model performance...")

        self.metrics = {
            "accuracy": float(accuracy_score(self.y_test, self.y_pred)),
            "precision": float(precision_score(self.y_test, self.y_pred)),
            "recall": float(recall_score(self.y_test, self.y_pred)),
            "f1_score": float(f1_score(self.y_test, self.y_pred)),
            "confusion_matrix": confusion_matrix(self.y_test, self.y_pred).tolist(),
        }

        if self.y_proba is not None:
            self.metrics["auc_roc"] = float(roc_auc_score(self.y_test, self.y_proba))
        else:
            self.metrics["auc_roc"] = None

        logger.success("✓ Model evaluation complete:")
        logger.info(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {self.metrics['f1_score']:.4f}")
        if self.metrics["auc_roc"]:
            logger.info(f"  AUC-ROC:   {self.metrics['auc_roc']:.4f}")

        return self

    def save_predictions(self, output_path: Optional[Path] = None) -> Path:
        """
        Save predictions to CSV

        Args:
            output_path: Path to save predictions

        Returns:
            Path to saved file
        """
        if self.y_pred is None:
            raise ValueError("No predictions to save")

        if output_path is None:
            output_path = self.config.paths.processed_data_dir / "test_predictions.csv"

        predictions_df = pd.DataFrame(
            {"true_label": self.y_test, "predicted_label": self.y_pred}
        )

        if self.y_proba is not None:
            predictions_df["probability"] = self.y_proba

        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)

        logger.success(f"✓ Predictions saved to: {output_path}")

        return output_path

    def save_metrics(self, output_path: Optional[Path] = None) -> Path:
        """
        Save metrics to JSON

        Args:
            output_path: Path to save metrics

        Returns:
            Path to saved file
        """
        if self.metrics is None:
            raise ValueError("No metrics to save. Call evaluate() first.")

        if output_path is None:
            output_path = self.config.paths.processed_data_dir / "test_metrics.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.success(f"✓ Metrics saved to: {output_path}")

        return output_path

    def print_classification_report(self):
        """Print detailed classification report"""
        if self.y_pred is None:
            raise ValueError("No predictions. Call predict() first.")

        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("=" * 60)

        report = classification_report(
            self.y_test, self.y_pred, target_names=["Bad Credit (0)", "Good Credit (1)"]
        )

        print(report)

    def get_metrics(self) -> Dict:
        """Get evaluation metrics dictionary"""
        if self.metrics is None:
            raise ValueError("No metrics. Call evaluate() first.")
        return self.metrics
