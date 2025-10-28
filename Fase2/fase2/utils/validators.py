"""
Data validation utilities
"""

import pandas as pd
from typing import List, Dict
from loguru import logger

from fase2.exceptions import DataValidationError


class DataValidator:
    """Validator for data quality checks"""

    @staticmethod
    def check_no_nulls(df: pd.DataFrame, columns: List[str] = None) -> bool:
        """
        Check that specified columns have no null values

        Args:
            df: DataFrame to check
            columns: List of columns to check. If None, checks all columns

        Returns:
            True if no nulls found, False otherwise

        Raises:
            DataValidationError: If nulls are found
        """
        cols_to_check = columns or df.columns.tolist()
        null_counts = df[cols_to_check].isnull().sum()

        if null_counts.sum() > 0:
            null_cols = null_counts[null_counts > 0]
            error_msg = f"Found null values in columns: {null_cols.to_dict()}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        return True

    @staticmethod
    def check_value_ranges(df: pd.DataFrame, ranges: Dict[str, List]) -> bool:
        """
        Check that categorical columns have values within expected ranges

        Args:
            df: DataFrame to check
            ranges: Dict mapping column names to lists of valid values

        Returns:
            True if all values are valid

        Raises:
            DataValidationError: If invalid values found
        """
        invalid_found = False

        for col, valid_values in ranges.items():
            if col not in df.columns:
                continue

            invalid_mask = ~df[col].isin(valid_values)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                invalid_vals = df[col][invalid_mask].unique()
                logger.error(
                    f"Column '{col}' has {invalid_count} invalid values: {invalid_vals}"
                )
                invalid_found = True

        if invalid_found:
            raise DataValidationError("Invalid categorical values found")

        return True

    @staticmethod
    def check_target_binary(df: pd.DataFrame, target_col: str) -> bool:
        """
        Check that target column is binary (0 or 1)

        Args:
            df: DataFrame to check
            target_col: Name of target column

        Returns:
            True if target is binary

        Raises:
            DataValidationError: If target is not binary
        """
        if target_col not in df.columns:
            raise DataValidationError(f"Target column '{target_col}' not found")

        unique_values = df[target_col].dropna().unique()
        valid_values = {0.0, 1.0}

        if not set(unique_values).issubset(valid_values):
            invalid = set(unique_values) - valid_values
            raise DataValidationError(
                f"Target column '{target_col}' has non-binary values: {invalid}"
            )

        return True

    @staticmethod
    def check_minimum_samples(df: pd.DataFrame, min_samples: int = 100) -> bool:
        """
        Check that dataset has minimum number of samples

        Args:
            df: DataFrame to check
            min_samples: Minimum required samples

        Returns:
            True if sufficient samples

        Raises:
            DataValidationError: If insufficient samples
        """
        n_samples = len(df)

        if n_samples < min_samples:
            raise DataValidationError(
                f"Insufficient samples: {n_samples} < {min_samples}"
            )

        logger.success(f"✓ Dataset has {n_samples} samples (>= {min_samples})")
        return True
