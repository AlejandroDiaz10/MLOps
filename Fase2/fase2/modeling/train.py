"""
Model training utilities (CLI wrapper)
"""

from pathlib import Path

from loguru import logger
import typer

from fase2.config import MODELS_DIR
from fase2.pipeline import MLPipeline

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option(
        "random_forest",
        help="Model to train (random_forest, logistic_regression, decision_tree)",
    ),
    output_dir: Path = typer.Option(MODELS_DIR, help="Directory to save trained model"),
):
    """
    Train machine learning models for credit risk prediction.

    This is a CLI wrapper around the MLPipeline.run_training() method.

    Supports: random_forest, logistic_regression, decision_tree

    Example:
        python -m fase2.modeling.train
        python -m fase2.modeling.train --model-name logistic_regression
    """
    try:
        pipeline = MLPipeline()
        pipeline.run_training(model_name=model_name, output_dir=output_dir)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
