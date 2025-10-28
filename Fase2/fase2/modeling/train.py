"""
Model training script using sklearn Pipeline.

This script follows Cookiecutter Data Science template and implements:
- Etapa 3: sklearn Pipeline with GridSearchCV
- Best practices for ML training
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from loguru import logger
from typing import Optional, Dict

from fase2.config import config
from fase2.pipeline_builder import PipelineBuilder
from fase2.core.model_factory import ModelFactory


def train_model(
    model_name: str = "random_forest",
    param_grid: Optional[Dict] = None,
    cv_folds: Optional[int] = None,
    save_model: bool = True,
) -> Path:
    """
    Train model using sklearn Pipeline with GridSearchCV.

    Args:
        model_name: Name of model to train
        param_grid: Custom parameter grid for GridSearch
        cv_folds: Number of cross-validation folds
        save_model: Whether to save trained pipeline

    Returns:
        Path to saved pipeline
    """
    logger.info("=" * 70)
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info("=" * 70)

    # Load training data
    logger.info("\n1️⃣ Loading processed data...")
    X_train_path = config.paths.processed_data_dir / "X_train.csv"
    y_train_path = config.paths.processed_data_dir / "y_train.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            "Training data not found. Run 'python -m fase2.features' first."
        )

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    logger.success(f"✓ Data loaded: {X_train.shape}")

    # Build sklearn Pipeline with GridSearch
    logger.info("\n2️⃣ Building sklearn Pipeline...")
    builder = PipelineBuilder(config)

    cv_folds = cv_folds or config.model.cv_folds

    grid_pipeline = builder.build_grid_search_pipeline(
        model_name=model_name, param_grid=param_grid, cv_folds=cv_folds
    )

    # Display pipeline structure
    if hasattr(grid_pipeline.estimator, "steps"):
        logger.info("\n📋 Pipeline Steps:")
        steps_df = builder.get_pipeline_steps(grid_pipeline.estimator)
        print(steps_df.to_string(index=False))

    # Train with GridSearchCV
    logger.info(f"\n3️⃣ Training with GridSearchCV ({cv_folds}-fold CV)...")
    grid_pipeline.fit(X_train, y_train)

    # Get best pipeline
    best_pipeline = grid_pipeline.best_estimator_
    best_score = grid_pipeline.best_score_
    best_params = grid_pipeline.best_params_

    logger.success("\n" + "=" * 70)
    logger.success("✓ TRAINING COMPLETE")
    logger.success(f"  Best CV AUC Score: {best_score:.4f}")
    logger.success(f"  Best Parameters: {best_params}")
    logger.success("=" * 70)

    # Save pipeline
    if save_model:
        logger.info("\n4️⃣ Saving pipeline...")

        output_dir = config.paths.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = model_name.replace(" ", "_").lower()
        pipeline_path = output_dir / f"{safe_name}_pipeline.pkl"

        joblib.dump(best_pipeline, pipeline_path)
        logger.success(f"✓ Pipeline saved to: {pipeline_path}")

        # Save metadata
        metadata = {
            "model_name": model_name,
            "pipeline_type": "sklearn_pipeline",
            "model_type": type(best_pipeline.named_steps["model"]).__name__,
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": {
                "n_samples": int(len(X_train)),
                "n_features": int(X_train.shape[1]),
                "class_distribution": {
                    "class_0": int((y_train == 0).sum()),
                    "class_1": int((y_train == 1).sum()),
                },
            },
            "grid_search": {
                "best_score": float(best_score),
                "best_params": best_params,
                "cv_folds": cv_folds,
            },
            "pipeline_steps": [
                {"name": name, "transformer": type(transformer).__name__}
                for name, transformer in best_pipeline.steps
            ],
        }

        metadata_path = output_dir / f"{safe_name}_pipeline_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"✓ Metadata saved to: {metadata_path}")

        return pipeline_path

    return None


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def train(
        model_name: str = typer.Option(
            "random_forest",
            "--model-name",
            "-m",
            help="Model to train: random_forest, logistic_regression, decision_tree",
        ),
        cv_folds: int = typer.Option(
            None,
            "--cv-folds",
            help="Number of cross-validation folds (default from config)",
        ),
    ):
        """
        Train ML model using sklearn Pipeline with GridSearchCV.

        Example:
            python -m fase2.modeling.train
            python -m fase2.modeling.train --model-name logistic_regression
            python -m fase2.modeling.train --cv-folds 3
        """
        train_model(model_name=model_name, cv_folds=cv_folds)

    app()


if __name__ == "__main__":
    main()
