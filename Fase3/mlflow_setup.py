"""
MLflow setup and initialization script.
"""

import mlflow
from loguru import logger
from fase3.config import config


def setup_mlflow():
    """Configure MLflow tracking server and create experiment."""

    # Set tracking URI
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    logger.info(f"MLflow tracking URI: {config.mlflow.tracking_uri}")

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            name=config.mlflow.experiment_name,
            artifact_location=config.mlflow.artifact_location,
        )
        logger.success(f"Created experiment: {config.mlflow.experiment_name}")
    except Exception as e:
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {config.mlflow.experiment_name}")

    mlflow.set_experiment(config.mlflow.experiment_name)
    return experiment_id


def test_mlflow_connection():
    """Test connection to MLflow server."""
    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        experiments = mlflow.search_experiments()
        logger.success(f"✅ MLflow connection OK. Found {len(experiments)} experiments")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {e}")
        return False


if __name__ == "__main__":
    test_mlflow_connection()
    setup_mlflow()

# python mlflow_setup.py
