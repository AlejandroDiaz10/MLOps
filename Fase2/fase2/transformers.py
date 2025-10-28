"""
Custom scikit-learn transformers for the preprocessing pipeline.
These transformers follow sklearn's API conventions (fit/transform).
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

from fase2.config import config


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove outliers using IQR method.

    NOTE: This transformer is designed for use OUTSIDE of pipelines
    or for marking outliers without removing them.

    For pipeline compatibility, set remove_rows=False to mark outliers
    instead of removing them.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        iqr_multiplier: float = 1.5,
        remove_rows: bool = False,
    ):  # 🆕 Nuevo parámetro
        """
        Initialize OutlierRemover.

        Args:
            columns: List of column names to check for outliers
            iqr_multiplier: IQR multiplier for outlier detection (default 1.5)
            remove_rows: Whether to remove outlier rows (False for pipeline compatibility)
        """
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.remove_rows = remove_rows
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        Compute outlier bounds on training data.

        Args:
            X: Training data
            y: Target (ignored)

        Returns:
            self
        """
        X_df = (
            pd.DataFrame(X, columns=self.columns)
            if not isinstance(X, pd.DataFrame)
            else X
        )

        columns_to_check = self.columns if self.columns else X_df.columns

        for col in columns_to_check:
            if col in X_df.columns:
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                self.bounds_[col] = (lower_bound, upper_bound)

        logger.debug(
            f"OutlierRemover fitted with bounds for {len(self.bounds_)} columns"
        )
        return self

    def transform(self, X):
        """
        Remove or clip outliers based on fitted bounds.

        Args:
            X: Data to transform

        Returns:
            Transformed data (rows removed if remove_rows=True, else clipped)
        """
        X_df = (
            pd.DataFrame(X, columns=self.columns)
            if not isinstance(X, pd.DataFrame)
            else X.copy()
        )

        if self.remove_rows:
            # Original behavior: remove rows (NOT pipeline compatible)
            mask = pd.Series([True] * len(X_df), index=X_df.index)

            for col, (lower, upper) in self.bounds_.items():
                if col in X_df.columns:
                    mask &= (X_df[col] >= lower) & (X_df[col] <= upper)

            X_clean = X_df[mask]
            n_removed = len(X_df) - len(X_clean)

            if n_removed > 0:
                logger.debug(
                    f"Removed {n_removed} outlier rows ({100*n_removed/len(X_df):.2f}%)"
                )

            return X_clean
        else:
            # 🆕 Pipeline-compatible: CLIP outliers instead of removing
            for col, (lower, upper) in self.bounds_.items():
                if col in X_df.columns:
                    # Clip values to bounds
                    X_df[col] = X_df[col].clip(lower=lower, upper=upper)

            logger.debug("Outliers clipped to bounds (pipeline-compatible mode)")
            return X_df


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from a DataFrame.

    Useful for separating numerical and categorical features in pipelines.
    """

    def __init__(self, columns: List[str]):
        """
        Initialize DataFrameSelector.

        Args:
            columns: List of column names to select
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Fit method (does nothing, required by sklearn API)."""
        return self

    def transform(self, X):
        """
        Select specified columns.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with selected columns
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X_df[self.columns]


class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Convert columns to numeric types.

    Handles non-numeric values by coercing them to NaN.
    """

    def __init__(self, exclude_columns: Optional[List[str]] = None):
        """
        Initialize TypeConverter.

        Args:
            exclude_columns: Columns to exclude from conversion
        """
        self.exclude_columns = exclude_columns or []

    def fit(self, X, y=None):
        """Fit method (does nothing, required by sklearn API)."""
        return self

    def transform(self, X):
        """
        Convert columns to numeric.

        Args:
            X: Input data

        Returns:
            Data with numeric types
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in X_df.columns:
            if col not in self.exclude_columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        return X_df


class CategoricalValidator(BaseEstimator, TransformerMixin):
    """
    Validate categorical variables against expected ranges.

    Removes rows with invalid categorical values.
    """

    def __init__(self, categorical_ranges: Optional[dict] = None):
        """
        Initialize CategoricalValidator.

        Args:
            categorical_ranges: Dict mapping column names to valid values
        """
        self.categorical_ranges = categorical_ranges or config.data.categorical_ranges

    def fit(self, X, y=None):
        """Fit method (does nothing, required by sklearn API)."""
        return self

    def transform(self, X):
        """
        Validate and filter categorical values.

        Args:
            X: Input data

        Returns:
            Data with only valid categorical values
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        mask = pd.Series([True] * len(X_df), index=X_df.index)

        for col, valid_values in self.categorical_ranges.items():
            if col in X_df.columns:
                mask &= X_df[col].isin(valid_values)

        X_clean = X_df[mask]
        n_removed = len(X_df) - len(X_clean)

        if n_removed > 0:
            logger.debug(
                f"CategoricalValidator removed {n_removed} rows with invalid values"
            )

        return X_clean


class TargetValidator(BaseEstimator, TransformerMixin):
    """
    Validate target variable (must be 0 or 1).

    Note: This transformer modifies both X and y, so should be used carefully.
    """

    def __init__(self, target_col: str = None):
        """
        Initialize TargetValidator.

        Args:
            target_col: Name of target column
        """
        self.target_col = target_col or config.data.target_col

    def fit(self, X, y=None):
        """Fit method (does nothing, required by sklearn API)."""
        return self

    def transform(self, X):
        """
        Validate target values (0 or 1 only).

        Args:
            X: Input data (must include target column)

        Returns:
            Data with valid target values only
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        if self.target_col not in X_df.columns:
            return X_df

        # Convert to numeric
        X_df[self.target_col] = pd.to_numeric(X_df[self.target_col], errors="coerce")

        # Filter valid values
        valid_mask = X_df[self.target_col].isin([0.0, 1.0])
        X_clean = X_df[valid_mask]

        n_removed = len(X_df) - len(X_clean)
        if n_removed > 0:
            logger.debug(
                f"TargetValidator removed {n_removed} rows with invalid target values"
            )

        return X_clean
