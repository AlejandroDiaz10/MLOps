"""
Model training and hyperparameter tuning
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
import joblib
import json

from fase2.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    AUC_THRESHOLD,
)

app = typer.Typer()


# ============= FUNCIONES REUTILIZABLES =============


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None
) -> RandomForestClassifier:
    """Train Random Forest with GridSearchCV"""
    logger.info("Training Random Forest Classifier...")

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Cross-validation folds: {CV_FOLDS}")

    grid_search = GridSearchCV(
        rf, param_grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"✓ Best parameters: {grid_search.best_params_}")
    logger.success(f"✓ Best CV AUC score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None
) -> LogisticRegression:
    """Train Logistic Regression with GridSearchCV"""
    logger.info("Training Logistic Regression...")

    if param_grid is None:
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        }

    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Cross-validation folds: {CV_FOLDS}")

    grid_search = GridSearchCV(
        lr, param_grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"✓ Best parameters: {grid_search.best_params_}")
    logger.success(f"✓ Best CV AUC score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None
) -> DecisionTreeClassifier:
    """Train Decision Tree with GridSearchCV"""
    logger.info("Training Decision Tree Classifier...")

    if param_grid is None:
        param_grid = {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Cross-validation folds: {CV_FOLDS}")

    grid_search = GridSearchCV(
        dt, param_grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"✓ Best parameters: {grid_search.best_params_}")
    logger.success(f"✓ Best CV AUC score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def save_model(
    model, model_name: str, metadata: dict = None, models_dir: Path = MODELS_DIR
) -> Path:
    """Save trained model and metadata"""
    logger.info(f"Saving model '{model_name}'...")

    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    safe_name = model_name.replace(" ", "_").lower()
    model_path = models_dir / f"{safe_name}.pkl"
    joblib.dump(model, model_path)
    logger.success(f"✓ Model saved to: {model_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = models_dir / f"{safe_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.success(f"✓ Metadata saved to: {metadata_path}")

    return model_path


# ============= CLI COMMAND =============


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_train.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    model_name: str = "random_forest",
    output_dir: Path = MODELS_DIR,
):
    """
    Train machine learning models for credit risk prediction.

    Supports: random_forest, logistic_regression, decision_tree
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading training data...")
    logger.info(f"  Features: {features_path}")
    logger.info(f"  Labels: {labels_path}")

    X_train = pd.read_csv(features_path)
    y_train = pd.read_csv(labels_path).values.ravel()

    logger.info(f"✓ Data loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"  Class distribution: {np.bincount(y_train.astype(int))}")

    # Train model based on selection
    if model_name == "random_forest":
        model = train_random_forest(X_train, y_train)
    elif model_name == "logistic_regression":
        model = train_logistic_regression(X_train, y_train)
    elif model_name == "decision_tree":
        model = train_decision_tree(X_train, y_train)
    else:
        logger.error(f"❌ Unknown model: {model_name}")
        logger.info(
            "Available models: random_forest, logistic_regression, decision_tree"
        )
        raise ValueError(f"Unknown model type: {model_name}")

    # Create metadata
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc")

    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "hyperparameters": model.get_params(),
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "n_samples": int(len(X_train)),
            "n_features": int(X_train.shape[1]),
            "class_distribution": {
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum()),
            },
        },
        "cross_validation": {
            "cv_folds": CV_FOLDS,
            "auc_scores": cv_scores.tolist(),
            "auc_mean": float(cv_scores.mean()),
            "auc_std": float(cv_scores.std()),
        },
        "meets_threshold": bool(cv_scores.mean() >= AUC_THRESHOLD),
        "threshold": AUC_THRESHOLD,
    }

    # Save model
    model_path = save_model(model, model_name, metadata, output_dir)

    logger.success("=" * 60)
    logger.success("✓ MODEL TRAINING COMPLETE")
    logger.success(f"✓ Model: {model_name}")
    logger.success(f"✓ CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.success(
        f"✓ Meets {AUC_THRESHOLD} threshold: {cv_scores.mean() >= AUC_THRESHOLD}"
    )
    logger.success("=" * 60)


if __name__ == "__main__":
    app()
