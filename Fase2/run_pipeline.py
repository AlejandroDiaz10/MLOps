#!/usr/bin/env python
"""
Main script to run the complete ML pipeline.

This is a convenience script for running the entire workflow.
"""
import typer
from loguru import logger

from fase2.pipeline import MLPipeline
from fase2.core.model_factory import ModelFactory

app = typer.Typer()


@app.command()
def full(
    model_name: str = typer.Option("random_forest", help="Model to train"),
    skip_data_prep: bool = typer.Option(False, help="Skip data preparation"),
    skip_feature_eng: bool = typer.Option(False, help="Skip feature engineering"),
):
    """
    Run the complete ML pipeline from start to finish.

    Example:
        python run_pipeline.py full
        python run_pipeline.py full --model-name logistic_regression
        python run_pipeline.py full --skip-data-prep --skip-feature-eng
    """
    pipeline = MLPipeline()
    pipeline.run_full_pipeline(
        model_name=model_name,
        skip_data_prep=skip_data_prep,
        skip_feature_eng=skip_feature_eng,
    )


@app.command()
def compare(
    skip_data_prep: bool = typer.Option(False, help="Skip data preparation"),
    skip_feature_eng: bool = typer.Option(False, help="Skip feature engineering"),
):
    """
    Train and compare all available models.

    Example:
        python run_pipeline.py compare
        python run_pipeline.py compare --skip-data-prep --skip-feature-eng
    """
    pipeline = MLPipeline()
    pipeline.run_multiple_models(
        skip_data_prep=skip_data_prep, skip_feature_eng=skip_feature_eng
    )


@app.command()
def models():
    """List all available model types."""
    available = ModelFactory.get_available_models()
    logger.info("Available models:")
    for model in available:
        print(f"  - {model}")


@app.command()
def sklearn(
    model_name: str = typer.Option("random_forest", help="Model to train"),
    skip_data_prep: bool = typer.Option(False, help="Skip data preparation"),
    skip_feature_eng: bool = typer.Option(False, help="Skip feature engineering"),
):
    """
    Run complete pipeline using sklearn Pipeline (BEST PRACTICE).

    This demonstrates industry best practices:
    - Single Pipeline object with all preprocessing
    - GridSearchCV for hyperparameter tuning
    - No data leakage
    - Fully reproducible

    Example:
        python run_pipeline.py sklearn
        python run_pipeline.py sklearn --model-name logistic_regression
        python run_pipeline.py sklearn --skip-data-prep --skip-feature-eng
    """
    pipeline = MLPipeline()
    pipeline.run_full_sklearn_pipeline(
        model_name=model_name,
        skip_data_prep=skip_data_prep,
        skip_feature_eng=skip_feature_eng,
        generate_plots=True,
    )


if __name__ == "__main__":
    app()
