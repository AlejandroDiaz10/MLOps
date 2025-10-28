"""
Data loading, cleaning, and preprocessing utilities (CLI wrapper)
"""

from pathlib import Path

from loguru import logger
import typer

from fase2.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from fase2.pipeline import MLPipeline

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "german_credit_modified.csv",
    output_path: Path = INTERIM_DATA_DIR / "german_credit_cleaned.csv",
):
    """
    Complete data cleaning pipeline for German Credit dataset.

    This is a CLI wrapper around the MLPipeline.run_data_preparation() method.

    Example:
        python -m fase2.dataset
        python -m fase2.dataset --input-path custom_data.csv
    """
    try:
        pipeline = MLPipeline()
        pipeline.run_data_preparation(input_path, output_path)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
