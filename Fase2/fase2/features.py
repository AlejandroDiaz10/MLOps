"""
Feature engineering and preprocessing utilities (CLI wrapper)
"""

from pathlib import Path

from loguru import logger
import typer

from fase2.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from fase2.pipeline import MLPipeline

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "german_credit_cleaned.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """
    Complete feature engineering pipeline.

    This is a CLI wrapper around the MLPipeline.run_feature_engineering() method.

    Reads cleaned data, handles outliers, splits data, scales features,
    and saves processed datasets ready for modeling.

    Example:
        python -m fase2.features
        python -m fase2.features --input-path custom_cleaned.csv
    """
    try:
        pipeline = MLPipeline()
        pipeline.run_feature_engineering(input_path, output_dir)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
