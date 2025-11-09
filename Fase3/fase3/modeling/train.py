"""
Model training script using sklearn Pipeline with MLflow tracking.

This script implements:
- sklearn Pipeline with preprocessing steps
- GridSearchCV for hyperparameter tuning
- MLflow experiment tracking with ALL metrics
- Model evaluation on test set
- Model versioning and registry
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

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


def evaluate_on_test_set(pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
    """
    Evaluate pipeline on test set and return all metrics.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("\n📊 Evaluating on test set...")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    if y_proba is not None:
        metrics["test_auc_roc"] = float(roc_auc_score(y_test, y_proba))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["test_confusion_matrix"] = cm.tolist()
    
    # Log metrics
    logger.info("  Test Set Metrics:")
    logger.info(f"    Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"    Precision: {metrics['test_precision']:.4f}")
    logger.info(f"    Recall:    {metrics['test_recall']:.4f}")
    logger.info(f"    F1-Score:  {metrics['test_f1_score']:.4f}")
    if "test_auc_roc" in metrics:
        logger.info(f"    AUC-ROC:   {metrics['test_auc_roc']:.4f}")
    
    return metrics


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
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            X_test_path = config.paths.processed_data_dir / "X_test.csv"
            y_test_path = config.paths.processed_data_dir / "y_test.csv"

            if not all([p.exists() for p in [X_train_path, y_train_path, X_test_path, y_test_path]]):
                error_msg = (
                    "Training/test data not found. Please run data preprocessing first:\n"
                    "  python -m fase3.dataset\n"
                    "  python -m fase3.features"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path).values.ravel()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()

            logger.success(f"✅ Data loaded successfully")
            logger.info(f"   Train shape: {X_train.shape}")
            logger.info(f"   Test shape: {X_test.shape}")

            # ========== LOG DATASET INFO ==========
            dataset_info = {
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_features": X_train.shape[1],
                "n_samples_class_0_train": int((y_train == 0).sum()),
                "n_samples_class_1_train": int((y_train == 1).sum()),
                "class_balance_ratio_train": float((y_train == 1).sum() / len(y_train)),
            }

            logger.info("\n📊 Dataset Statistics:")
            logger.info(f"   Train samples: {dataset_info['n_samples_train']}")
            logger.info(f"   Test samples: {dataset_info['n_samples_test']}")
            logger.info(f"   Features: {dataset_info['n_features']}")
            logger.info(f"   Train Class 0: {dataset_info['n_samples_class_0_train']}")
            logger.info(f"   Train Class 1: {dataset_info['n_samples_class_1_train']}")

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

            # ========== TRAIN MODEL ==========
            logger.info(f"\n3️⃣ Training with GridSearchCV ({cv_folds}-fold CV)...")

            import time
            start_time = time.time()

            grid_pipeline.fit(X_train, y_train)

            training_time = time.time() - start_time
            logger.success(f"✅ Training completed in {training_time:.2f} seconds")

            # ========== GET BEST RESULTS ==========
            best_pipeline = grid_pipeline.best_estimator_
            best_cv_score = grid_pipeline.best_score_
            best_params = grid_pipeline.best_params_

            logger.success("\n" + "=" * 70)
            logger.success("🎯 CROSS-VALIDATION RESULTS")
            logger.success("=" * 70)
            logger.info(f"Best CV Score (AUC-ROC): {best_cv_score:.4f}")
            logger.info(f"Best Parameters: {best_params}")

            # ========== EVALUATE ON TEST SET ==========
            test_metrics = evaluate_on_test_set(best_pipeline, X_test, y_test)

            # ========== LOG METRICS TO MLFLOW ==========
            if mlflow_enabled:
                try:
                    # Log best params
                    mlflow.log_params(best_params)

                    # Log CV metrics
                    mlflow.log_metric("cv_best_score", best_cv_score)
                    mlflow.log_metric("training_time_seconds", training_time)

                    # Log ALL test metrics
                    for metric_name, metric_value in test_metrics.items():
                        if metric_name != "test_confusion_matrix":  # Skip matrix
                            mlflow.log_metric(metric_name, metric_value)

                except Exception as e:
                    logger.warning(f"⚠️ Failed to log metrics to MLflow: {e}")

            # ========== LOG MODEL TO MLFLOW ==========
            if mlflow_enabled:
                try:
                    logger.info("\n4️⃣ Logging model to MLflow...")

                    signature = infer_signature(X_train, best_pipeline.predict(X_train))

                    # Register with unique name per model type
                    registered_model_name = f"{config.mlflow.experiment_name}_{model_name}"

                    mlflow.sklearn.log_model(
                        best_pipeline,
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=registered_model_name,
                        input_example=X_train.head(1),
                    )

                    logger.success(f"✅ Model logged to MLflow registry as: {registered_model_name}")

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
                    "mlflow_experiment": config.mlflow.experiment_name if mlflow_enabled else None,
                    "dataset_info": dataset_info,
                    "training_params": training_params,
                    "grid_search": {
                        "best_cv_score": float(best_cv_score),
                        "best_params": best_params,
                        "cv_folds": cv_folds,
                        "training_time_seconds": training_time,
                    },
                    "test_metrics": test_metrics,
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

            # ========== SAVE METRICS FOR DVC ==========
            metrics_output = {
                "cv_best_score": float(best_cv_score),
                "training_time_seconds": float(training_time),
                "test_accuracy": test_metrics["test_accuracy"],
                "test_precision": test_metrics["test_precision"],
                "test_recall": test_metrics["test_recall"],
                "test_f1_score": test_metrics["test_f1_score"],
                "test_auc_roc": test_metrics.get("test_auc_roc", 0.0),
                "n_samples_train": int(len(X_train)),
                "n_samples_test": int(len(X_test)),
                "n_features": int(X_train.shape[1]),
            }

            metrics_path = config.paths.proj_root / "reports" / "metrics" / f"{model_name}_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, "w") as f:
                json.dump(metrics_output, f, indent=2)

            logger.info(f"✅ Metrics saved for DVC: {metrics_path}")

            # ========== FINAL SUMMARY ==========
            logger.info("\n" + "🎉" * 35)
            logger.success("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("🎉" * 35)

            if mlflow_enabled:
                logger.info(f"\n📊 MLflow:")
                logger.info(f"   Run ID: {run_id}")
                logger.info(f"   Tracking URI: {config.mlflow.tracking_uri}")

            if pipeline_path:
                logger.info(f"\n💾 Saved Files:")
                logger.info(f"   Pipeline: {pipeline_path}")
                logger.info(f"   Metadata: {metadata_path}")
                logger.info(f"   Metrics: {metrics_path}")

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
