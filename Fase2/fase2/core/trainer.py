"""
Model training class with GridSearchCV and cross-validation
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
import joblib
import json

from fase2.config import config
from fase2.core.model_factory import ModelFactory
from fase2.exceptions import ModelTrainingError


class ModelTrainer:
    """
    Handles model training with hyperparameter tuning via GridSearchCV.

    Attributes:
        config: Configuration object
        X_train, y_train: Training data
        model: Trained model instance
        grid_search: GridSearchCV object with results
        cv_scores: Cross-validation scores
    """

    def __init__(self, config_obj=None):
        """Initialize ModelTrainer"""
        self.config = config_obj or config
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.factory = ModelFactory()
        logger.debug("ModelTrainer initialized")

    def load_training_data(
        self, X_train_path: Optional[Path] = None, y_train_path: Optional[Path] = None
    ) -> "ModelTrainer":
        """
        Load training data from CSV files

        Args:
            X_train_path: Path to training features
            y_train_path: Path to training labels

        Returns:
            self: For method chaining
        """
        if X_train_path is None:
            X_train_path = self.config.paths.processed_data_dir / "X_train.csv"
        if y_train_path is None:
            y_train_path = self.config.paths.processed_data_dir / "y_train.csv"

        logger.info("Loading training data...")
        logger.info(f"  Features: {X_train_path}")
        logger.info(f"  Labels: {y_train_path}")

        self.X_train = pd.read_csv(X_train_path)
        self.y_train = pd.read_csv(y_train_path).values.ravel()

        logger.success(
            f"✓ Data loaded: X_train {self.X_train.shape}, y_train {self.y_train.shape}"
        )
        logger.info(f"  Class distribution: {np.bincount(self.y_train.astype(int))}")

        return self

    def train(
        self,
        model_name: str,
        param_grid: Optional[Dict] = None,
        cv_folds: Optional[int] = None,
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> "ModelTrainer":
        """
        Train model with GridSearchCV for hyperparameter tuning

        Args:
            model_name: Name of model to train
            param_grid: Custom parameter grid (uses default if None)
            cv_folds: Number of CV folds (uses config if None)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level for GridSearchCV

        Returns:
            self: For method chaining
        """
        if self.X_train is None or self.y_train is None:
            raise ModelTrainingError("Must load training data first")

        self.model_name = model_name
        cv_folds = cv_folds or self.config.model.cv_folds

        logger.info("=" * 60)
        logger.info(f"Training {model_name.upper()} with GridSearchCV")
        logger.info("=" * 60)

        # Create base model
        base_model = self.factory.create_model(model_name)

        # Get parameter grid
        param_grid = self.factory.get_param_grid(model_name, param_grid)
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Cross-validation folds: {cv_folds}")

        # GridSearchCV
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=n_jobs,
            verbose=verbose,
        )

        logger.info("Starting grid search...")
        self.grid_search.fit(self.X_train, self.y_train)

        # Get best model
        self.model = self.grid_search.best_estimator_

        logger.success("=" * 60)
        logger.success("✓ Training complete!")
        logger.success(f"  Best parameters: {self.grid_search.best_params_}")
        logger.success(f"  Best CV AUC score: {self.grid_search.best_score_:.4f}")
        logger.success("=" * 60)

        return self

    def evaluate(self, cv_folds: Optional[int] = None) -> "ModelTrainer":
        """
        Evaluate model with cross-validation

        Args:
            cv_folds: Number of CV folds

        Returns:
            self: For method chaining
        """
        if self.model is None:
            raise ModelTrainingError("Must train model first")

        cv_folds = cv_folds or self.config.model.cv_folds

        logger.info(f"Evaluating model with {cv_folds}-fold cross-validation...")

        self.cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv_folds, scoring="roc_auc"
        )

        mean_score = self.cv_scores.mean()
        std_score = self.cv_scores.std()

        logger.success(f"✓ CV AUC: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"  Individual fold scores: {[f'{s:.4f}' for s in self.cv_scores]}")

        # Check threshold
        threshold = self.config.model.auc_threshold
        meets_threshold = mean_score >= threshold

        if meets_threshold:
            logger.success(f"✓ Meets {threshold} threshold!")
        else:
            logger.warning(
                f"⚠ Does not meet {threshold} threshold ({mean_score:.4f} < {threshold})"
            )

        return self

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save trained model and metadata

        Args:
            output_dir: Directory to save model

        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ModelTrainingError("No model to save. Train a model first.")

        if output_dir is None:
            output_dir = self.config.paths.models_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        safe_name = self.model_name.replace(" ", "_").lower()
        model_path = output_dir / f"{safe_name}.pkl"
        joblib.dump(self.model, model_path)
        logger.success(f"✓ Model saved to: {model_path}")

        # Save metadata
        metadata = self._create_metadata()
        metadata_path = output_dir / f"{safe_name}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.success(f"✓ Metadata saved to: {metadata_path}")

        return model_path

    def _create_metadata(self) -> Dict:
        """Create metadata dictionary for model"""
        metadata = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "hyperparameters": self.model.get_params(),
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "n_samples": int(len(self.X_train)),
                "n_features": int(self.X_train.shape[1]),
                "class_distribution": {
                    "class_0": int((self.y_train == 0).sum()),
                    "class_1": int((self.y_train == 1).sum()),
                },
            },
        }

        if self.grid_search is not None:
            metadata["grid_search"] = {
                "best_params": self.grid_search.best_params_,
                "best_score": float(self.grid_search.best_score_),
                "n_splits": self.grid_search.n_splits_,
            }

        if self.cv_scores is not None:
            metadata["cross_validation"] = {
                "cv_folds": len(self.cv_scores),
                "auc_scores": [float(s) for s in self.cv_scores],
                "auc_mean": float(self.cv_scores.mean()),
                "auc_std": float(self.cv_scores.std()),
                "meets_threshold": bool(
                    self.cv_scores.mean() >= self.config.model.auc_threshold
                ),
                "threshold": self.config.model.auc_threshold,
            }

        return metadata

    def get_model(self):
        """Get the trained model"""
        if self.model is None:
            raise ModelTrainingError("No model trained yet")
        return self.model
