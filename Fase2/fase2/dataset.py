"""
Data cleaning and preparation script.

This script follows Cookiecutter Data Science template and implements:
- Data loading from raw sources
- Data cleaning and validation
- Saves cleaned data to interim directory
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from fase2.config import config
from fase2.core.data_processor import DataProcessor


def clean_data(input_path: Path = None, output_path: Path = None) -> Path:
    """
    Clean raw data and save to interim directory.

    This function orchestrates the DataProcessor class to:
    1. Load raw data
    2. Translate columns to English
    3. Clean whitespace
    4. Convert to numeric types
    5. Validate target variable
    6. Handle missing values
    7. Validate categorical ranges
    8. Remove duplicates

    Args:
        input_path: Path to raw data file (default from config)
        output_path: Path to save cleaned data (default from config)

    Returns:
        Path to cleaned data file
    """
    logger.info("Starting data cleaning process...")

    # Default paths
    if input_path is None:
        input_path = config.paths.raw_data_dir / "german_credit_modified.csv"

    if output_path is None:
        output_path = config.paths.interim_data_dir / "german_credit_cleaned.csv"

    # Use DataProcessor with method chaining
    processor = DataProcessor()

    df_clean = (
        processor.load_raw_data(input_path)
        .translate_columns()
        .clean_whitespace()
        .convert_to_numeric()
        .validate_target()
        .handle_missing_values()
        .validate_categorical_ranges()
        .remove_duplicates()
        .get_data()
    )

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    logger.success(f"✓ Cleaned data saved to: {output_path}")
    logger.info(f"  Final shape: {df_clean.shape}")

    return output_path


def main():
    """CLI entry point for data cleaning."""
    try:
        clean_data()
        logger.success("✓ Data cleaning completed successfully")
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise


if __name__ == "__main__":
    main()
