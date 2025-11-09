"""
Model training script using sklearn Pipeline with MLflow tracking.

This script implements:
- sklearn Pipeline with preprocessing steps
- GridSearchCV for hyperparameter tuning
- MLflow experiment tracking
- Model versioning and registry
- Comprehensive logging and error handling
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from loguru import logger
from typing import Optional, Dict
from datetime import datetime

from fase3.config import config
from fase3.pipeline_builder import PipelineBuilder
from fase3.core.model_factory import ModelFactory


def setup_mlflow() -> None:
    """
    Configure MLflow tracking server and experiment.

    Raises:
        Exception: If MLflow server is unreachable
    """
    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        logger.info(f"🔗 MLflow tracking URI: {config.mlflow.tracking_uri}")

        # Test connection
        mlflow.search_experiments()
        logger.success("✅ MLflow connection successful")

        # Set experiment
        mlflow.set_experiment(config.mlflow.experiment_name)
        logger.info(f"📊 Using experiment: {config.mlflow.experiment_name}")

    except Exception as e:
        logger.error(f"❌ Failed to connect to MLflow: {e}")
        logger.warning("⚠️ Continuing without MLflow tracking...")
        raise


def train_model(
    model_name: str = "random_forest",
    param_grid: Optional[Dict] = None,
    cv_folds: Optional[int] = None,
    save_model: bool = True,
    use_mlflow: bool = True,
) -> Path:
    """
    Train model using sklearn Pipeline with GridSearchCV and MLflow tracking.

    Args:
        model_name: Name of the model to train (random_forest, logistic_regression, decision_tree)
        param_grid: Custom hyperparameter grid for GridSearch (uses defaults if None)
        cv_folds: Number of cross-validation folds (uses config default if None)
        save_model: Whether to save the trained pipeline locally
        use_mlflow: Whether to use MLflow tracking (can disable for local testing)

    Returns:
        Path to the saved pipeline file

    Raises:
        FileNotFoundError: If training data is not found
        Exception: If training fails

    Example:
        >>> model_path = train_model(model_name="random_forest", cv_folds=5)
        >>> print(f"Model saved to: {model_path}")
    """

    logger.info("=" * 70)
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info("=" * 70)

    # ========== SETUP MLFLOW ==========
    mlflow_enabled = use_mlflow
    run_id = None

    if use_mlflow:
        try:
            setup_mlflow()
        except Exception as e:
            logger.warning("⚠️ MLflow setup failed, continuing without tracking")
            mlflow_enabled = False

    # Start MLflow run context
    mlflow_context = (
        mlflow.start_run(
            run_name=f"{model_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if mlflow_enabled
        else NoOpContext()
    )

    with mlflow_context:
        try:
            # Get run ID if MLflow is enabled
            if mlflow_enabled:
                run_id = mlflow.active_run().info.run_id
                logger.info(f"🔖 MLflow Run ID: {run_id}")

            # ========== LOG BASIC PARAMETERS ==========
            training_params = {
                "model_name": model_name,
                "cv_folds": cv_folds or config.model.cv_folds,
                "test_size": config.model.test_size,
                "random_state": config.model.random_state,
                "timestamp": datetime.now().isoformat(),
            }

            if mlflow_enabled:
                try:
                    for key, value in training_params.items():
                        mlflow.log_param(key, value)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log params to MLflow: {e}")

            # ========== LOAD TRAINING DATA ==========
            logger.info("\n1️⃣ Loading processed data...")
            X_train_path = config.paths.processed_data_dir / "X_train.csv"
            y_train_path = config.paths.processed_data_dir / "y_train.csv"

            if not X_train_path.exists() or not y_train_path.exists():
                error_msg = (
                    "Training data not found. Please run data preprocessing first:\n"
                    "  python -m fase3.dataset\n"
                    "  python -m fase3.features"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path).values.ravel()

            logger.success(f"✅ Data loaded successfully")
            logger.info(f"   Shape: {X_train.shape}")
            logger.info(
                f"   Features: {list(X_train.columns[:5])}... ({len(X_train.columns)} total)"
            )

            # ========== LOG DATASET INFO ==========
            dataset_info = {
                "n_samples_train": len(X_train),
                "n_features": X_train.shape[1],
                "n_samples_class_0": int((y_train == 0).sum()),
                "n_samples_class_1": int((y_train == 1).sum()),
                "class_balance_ratio": float((y_train == 1).sum() / len(y_train)),
            }

            logger.info("\n📊 Dataset Statistics:")
            logger.info(f"   Total samples: {dataset_info['n_samples_train']}")
            logger.info(f"   Features: {dataset_info['n_features']}")
            logger.info(f"   Class 0 (Bad Credit): {dataset_info['n_samples_class_0']}")
            logger.info(
                f"   Class 1 (Good Credit): {dataset_info['n_samples_class_1']}"
            )
            logger.info(f"   Class balance: {dataset_info['class_balance_ratio']:.2%}")

            if mlflow_enabled:
                try:
                    for key, value in dataset_info.items():
                        mlflow.log_param(key, value)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log dataset info: {e}")

            # ========== BUILD PIPELINE ==========
            logger.info("\n2️⃣ Building sklearn Pipeline with GridSearchCV...")
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

            # Log parameter grid
            if mlflow_enabled and param_grid:
                try:
                    for key, value in param_grid.items():
                        mlflow.log_param(f"grid_{key}", str(value))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log param grid: {e}")

            # ========== TRAIN MODEL ==========
            logger.info(f"\n3️⃣ Training with GridSearchCV ({cv_folds}-fold CV)...")
            logger.info("   This may take a few minutes...")

            import time

            start_time = time.time()

            grid_pipeline.fit(X_train, y_train)

            training_time = time.time() - start_time
            logger.success(f"✅ Training completed in {training_time:.2f} seconds")

            # ========== GET BEST RESULTS ==========
            best_pipeline = grid_pipeline.best_estimator_
            best_score = grid_pipeline.best_score_
            best_params = grid_pipeline.best_params_

            logger.success("\n" + "=" * 70)
            logger.success("🎯 TRAINING RESULTS")
            logger.success("=" * 70)
            logger.success(f"✅ Best CV AUC-ROC Score: {best_score:.4f}")
            logger.success(f"✅ Training Time: {training_time:.2f}s")
            logger.info("\n🔧 Best Hyperparameters:")
            for key, value in best_params.items():
                logger.info(f"   {key}: {value}")
            logger.success("=" * 70)

            # ========== LOG METRICS TO MLFLOW ==========
            if mlflow_enabled:
                try:
                    mlflow.log_metric("cv_best_score", best_score)
                    mlflow.log_metric("training_time_seconds", training_time)
                    mlflow.log_params(best_params)

                    # Log CV results
                    cv_results = grid_pipeline.cv_results_
                    mlflow.log_metric(
                        "cv_mean_score", cv_results["mean_test_score"].max()
                    )
                    mlflow.log_metric(
                        "cv_std_score",
                        cv_results["std_test_score"][
                            cv_results["mean_test_score"].argmax()
                        ],
                    )

                except Exception as e:
                    logger.warning(f"⚠️ Failed to log metrics to MLflow: {e}")

            # ========== LOG MODEL TO MLFLOW ==========
            if mlflow_enabled:
                try:
                    logger.info("\n4️⃣ Logging model to MLflow...")

                    signature = infer_signature(X_train, best_pipeline.predict(X_train))

                    mlflow.sklearn.log_model(
                        best_pipeline,
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=f"{model_name}_classifier",
                        input_example=X_train.head(1),
                    )

                    logger.success("✅ Model logged to MLflow registry")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to log model to MLflow: {e}")

            # ========== SAVE MODEL LOCALLY ==========
            pipeline_path = None

            if save_model:
                logger.info("\n5️⃣ Saving pipeline locally...")

                output_dir = config.paths.models_dir
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_name = model_name.replace(" ", "_").lower()
                pipeline_path = output_dir / f"{safe_name}_pipeline.pkl"

                # Save pipeline
                joblib.dump(best_pipeline, pipeline_path)
                logger.success(f"✅ Pipeline saved to: {pipeline_path}")

                # Save metadata
                metadata = {
                    "model_name": model_name,
                    "pipeline_type": "sklearn_pipeline",
                    "model_type": type(best_pipeline.named_steps["model"]).__name__,
                    "training_date": datetime.now().isoformat(),
                    "mlflow_run_id": run_id,
                    "mlflow_experiment": (
                        config.mlflow.experiment_name if mlflow_enabled else None
                    ),
                    "dataset_info": dataset_info,
                    "training_params": training_params,
                    "grid_search": {
                        "best_score": float(best_score),
                        "best_params": best_params,
                        "cv_folds": cv_folds,
                        "training_time_seconds": training_time,
                    },
                    "pipeline_steps": [
                        {"name": name, "transformer": type(transformer).__name__}
                        for name, transformer in best_pipeline.steps
                    ],
                    "feature_names": list(X_train.columns),
                }

                metadata_path = output_dir / f"{safe_name}_pipeline_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.success(f"✅ Metadata saved to: {metadata_path}")

                # Log artifacts to MLflow
                if mlflow_enabled:
                    try:
                        mlflow.log_artifact(str(pipeline_path))
                        mlflow.log_artifact(str(metadata_path))
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to log artifacts to MLflow: {e}")

            # ========== FINAL SUMMARY ==========
            logger.info("\n" + "🎉" * 35)
            logger.success("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("🎉" * 35)

            if mlflow_enabled:
                logger.info(f"\n📊 MLflow:")
                logger.info(f"   Run ID: {run_id}")
                logger.info(f"   Tracking URI: {config.mlflow.tracking_uri}")
                logger.info(
                    f"   View in UI: {config.mlflow.tracking_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}"
                )

            if pipeline_path:
                logger.info(f"\n💾 Saved Files:")
                logger.info(f"   Pipeline: {pipeline_path}")
                logger.info(f"   Metadata: {metadata_path}")

            # ========== SAVE METRICS FOR DVC ==========
            metrics_output = {
                "cv_best_score": float(best_score),
                "training_time_seconds": float(training_time),
                "n_samples_train": int(len(X_train)),
                "n_features": int(X_train.shape[1]),
            }

            metrics_path = (
                config.paths.proj_root / "reports" / "metrics" / "train_metrics.json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(metrics_output, f, indent=2)

            logger.info(f"✅ Metrics saved for DVC: {metrics_path}")

            return pipeline_path

        except Exception as e:
            logger.error("\n" + "❌" * 35)
            logger.error(f"❌ TRAINING FAILED: {str(e)}")
            logger.error("❌" * 35)
            logger.exception("Full error traceback:")

            # Log failure to MLflow
            if mlflow_enabled:
                try:
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error_message", str(e))
                except:
                    pass

            raise


class NoOpContext:
    """No-op context manager for when MLflow is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer(help="Train ML models with sklearn Pipeline and MLflow tracking")

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
        no_mlflow: bool = typer.Option(
            False,
            "--no-mlflow",
            help="Disable MLflow tracking (for local testing)",
        ),
        no_save: bool = typer.Option(
            False,
            "--no-save",
            help="Don't save model locally (only to MLflow)",
        ),
    ):
        """
        Train ML model using sklearn Pipeline with GridSearchCV and MLflow.

        Examples:
            # Train Random Forest with default settings
            python -m fase3.modeling.train

            # Train Logistic Regression with 3-fold CV
            python -m fase3.modeling.train --model-name logistic_regression --cv-folds 3

            # Train without MLflow (local testing)
            python -m fase3.modeling.train --no-mlflow

            # Train and save only to MLflow
            python -m fase3.modeling.train --no-save
        """
        try:
            train_model(
                model_name=model_name,
                cv_folds=cv_folds,
                use_mlflow=not no_mlflow,
                save_model=not no_save,
            )
            logger.success("\n✅ Training script completed successfully!")

        except Exception as e:
            logger.error(f"\n❌ Training script failed: {e}")
            raise typer.Exit(code=1)

    app()


if __name__ == "__main__":
    main()
