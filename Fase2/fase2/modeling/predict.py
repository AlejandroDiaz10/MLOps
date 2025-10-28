"""
Model inference and prediction utilities (CLI wrapper)
"""

from pathlib import Path

from loguru import logger
import typer

from fase2.config import MODELS_DIR
from fase2.pipeline import MLPipeline

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(
        MODELS_DIR / "random_forest.pkl", help="Path to trained model"
    ),
    save_results: bool = typer.Option(True, help="Save predictions and metrics"),
):
    """
    Perform inference using a trained model.

    This is a CLI wrapper around the MLPipeline.run_evaluation() method.

    Loads test data, makes predictions, and evaluates performance.

    Example:
        python -m fase2.modeling.predict
        python -m fase2.modeling.predict --model-path models/logistic_regression.pkl
    """
    try:
        pipeline = MLPipeline()
        pipeline.run_evaluation(model_path=model_path, save_results=save_results)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
