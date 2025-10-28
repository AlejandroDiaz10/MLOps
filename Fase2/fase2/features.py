"""
Feature engineering script.

This script follows Cookiecutter Data Science template and implements:
- Loading cleaned data
- Outlier detection and removal
- Train-test split
- Feature scaling
- Saves processed data ready for modeling
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from fase2.config import config
from fase2.core.feature_engineer import FeatureEngineer


def create_features(input_path: Path = None, output_dir: Path = None) -> dict:
    """
    Create features from cleaned data.

    This function orchestrates the FeatureEngineer class to:
    1. Load cleaned data
    2. Detect and handle outliers
    3. Split features and target
    4. Create train-test split
    5. Scale features
    6. Save processed data

    Args:
        input_path: Path to cleaned data (default from config)
        output_dir: Directory to save processed data (default from config)

    Returns:
        Dictionary with paths to saved files
    """
    logger.info("Starting feature engineering process...")

    # Default paths
    if input_path is None:
        input_path = config.paths.interim_data_dir / "german_credit_cleaned.csv"

    if output_dir is None:
        output_dir = config.paths.processed_data_dir

    # Use FeatureEngineer with method chaining
    engineer = FeatureEngineer()

    paths = (
        engineer.load_data(input_path)
        .detect_outliers()
        .split_target()
        .train_test_split()
        .scale_features()
        .save_all(output_dir)
    )

    logger.success("✓ Feature engineering completed")
    logger.info("  Files saved:")
    for name, path in paths.items():
        logger.info(f"    {name}: {path}")

    return paths


def main():
    """CLI entry point for feature engineering."""
    try:
        create_features()
        logger.success("✓ Feature engineering completed successfully")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
