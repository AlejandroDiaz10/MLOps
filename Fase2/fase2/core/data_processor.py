"""
Data processing class with OOP principles
"""

from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
from loguru import logger

from fase2.config import config


class DataProcessor:
    """
    Handles all data loading, cleaning, and validation operations.
    
    Uses method chaining for clean API:
        processor = DataProcessor()
        df = processor.load_raw_data() \\
                      .translate_columns() \\
                      .clean_whitespace() \\
                      .get_data()
    
    Attributes:
        config: Configuration object
        df: Current DataFrame being processed
    """

    def __init__(self, config_obj=None):
        """
        Initialize DataProcessor with configuration

        Args:
            config_obj: Optional custom configuration. Uses global config if None.
        """
        self.config = config_obj or config
        self.df: Optional[pd.DataFrame] = None
        self._original_shape: Optional[Tuple[int, int]] = None
        logger.debug("DataProcessor initialized")

    def load_raw_data(self, filepath: Optional[Path] = None) -> "DataProcessor":
        """
        Load raw data from CSV file.

        Args:
            filepath: Path to CSV file. If None, uses default from config.

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If file cannot be loaded
            ValueError: If data cannot be parsed
        """
        if filepath is None:
            filepath = self.config.paths.raw_data_dir / "german_credit_modified.csv"

        try:
            logger.info(f"Loading data from: {filepath}")
            self.df = pd.read_csv(filepath)
            self._original_shape = self.df.shape
            logger.success(
                f"✓ Data loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns"
            )
            return self
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {filepath}: {str(e)}")

    def translate_columns(self) -> "DataProcessor":
        """
        Translate German column names to English

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info("Translating column names from German to English...")

        self.df = self.df.rename(columns=self.config.data.column_mapping)

        translated_count = sum(
            1
            for col in self.df.columns
            if col in self.config.data.column_mapping.values()
        )
        logger.success(f"✓ Translated {translated_count} column names")

        return self

    def clean_whitespace(self) -> "DataProcessor":
        """
        Remove whitespace from all string columns and drop problematic columns

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info("Cleaning whitespace from all columns...")

        # Remove problematic column if exists
        if "mixed_type_col" in self.df.columns:
            self.df = self.df.drop(columns=["mixed_type_col"])
            logger.info("  ✓ Dropped 'mixed_type_col'")

        # Strip whitespace from object columns
        object_cols = self.df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            self.df[col] = self.df[col].astype(str).str.strip()

        logger.success(f"✓ Whitespace removed from {len(object_cols)} columns")
        return self

    def convert_to_numeric(self) -> "DataProcessor":
        """
        Convert all non-target columns to numeric types

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info("Converting columns to numeric types...")

        numeric_cols = [
            col for col in self.df.columns if col != self.config.data.target_col
        ]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        logger.success(f"✓ Converted {len(numeric_cols)} columns to numeric")
        return self

    def validate_target(self) -> "DataProcessor":
        """
        Validate target variable (must be 0 or 1)

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info(f"Validating target variable '{self.config.data.target_col}'...")

        target_col = self.config.data.target_col
        rows_before = len(self.df)

        # Show unique values before cleaning
        unique_vals = self.df[target_col].value_counts(dropna=False)
        logger.debug(f"Target unique values: {unique_vals.to_dict()}")

        # Convert to numeric
        self.df[target_col] = pd.to_numeric(self.df[target_col], errors="coerce")

        # Filter valid values (0.0 or 1.0)
        valid_mask = self.df[target_col].isin([0.0, 1.0])
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid target values, removing...")
            self.df = self.df[valid_mask]

        # Remove NaNs
        nan_count = self.df[target_col].isnull().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN targets, removing...")
            self.df = self.df.dropna(subset=[target_col])

        rows_removed = rows_before - len(self.df)
        logger.success(f"✓ Target validated. Removed {rows_removed} invalid rows")

        return self

    def handle_missing_values(self) -> "DataProcessor":
        """
        Handle missing values with appropriate imputation strategy:
        - Continuous features: median imputation
        - Discrete features: mode imputation

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info("Handling missing values...")

        missing_count = self.df.isnull().sum().sum()
        if missing_count == 0:
            logger.success("✓ No missing values found")
            return self

        logger.info(f"Found {missing_count} missing values")

        # Impute continuous with median
        imputed_count = 0
        for col in self.config.data.continuous_features:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                n_imputed = self.df[col].isnull().sum()
                self.df[col] = self.df[col].fillna(median_val)
                imputed_count += n_imputed
                logger.debug(
                    f"  [{col}] Imputed {n_imputed} values with median ({median_val:.2f})"
                )

        # Impute discrete with mode
        discrete_cols = [
            col
            for col in self.df.columns
            if col not in self.config.data.continuous_features
            and col != self.config.data.target_col
        ]

        for col in discrete_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 0
                n_imputed = self.df[col].isnull().sum()
                self.df[col] = self.df[col].fillna(mode_val)
                imputed_count += n_imputed
                logger.debug(
                    f"  [{col}] Imputed {n_imputed} values with mode ({mode_val})"
                )

        # Drop any remaining NaNs
        rows_before = len(self.df)
        self.df = self.df.dropna()
        rows_dropped = rows_before - len(self.df)

        if rows_dropped > 0:
            logger.warning(f"  Dropped {rows_dropped} rows with remaining NaNs")

        logger.success(
            f"✓ Missing values handled. Imputed: {imputed_count}, Dropped: {rows_dropped}"
        )
        return self

    def validate_categorical_ranges(self) -> "DataProcessor":
        """
        Validate categorical variables against expected ranges

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()
        logger.info("Validating categorical variable ranges...")

        rows_before = len(self.df)
        invalid_total = 0

        for col, valid_values in self.config.data.categorical_ranges.items():
            if col not in self.df.columns:
                continue

            invalid_mask = ~self.df[col].isin(valid_values)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                logger.debug(f"  [{col}] Found {invalid_count} invalid values")
                self.df = self.df[~invalid_mask]
                invalid_total += invalid_count

        rows_removed = rows_before - len(self.df)

        if rows_removed > 0:
            logger.success(
                f"✓ Removed {rows_removed} rows with invalid categorical values"
            )
        else:
            logger.success("✓ All categorical variables valid")

        return self

    def remove_duplicates(self) -> "DataProcessor":
        """
        Remove duplicate rows

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()

        rows_before = len(self.df)
        self.df = self.df.drop_duplicates()
        rows_removed = rows_before - len(self.df)

        if rows_removed > 0:
            logger.info(f"✓ Removed {rows_removed} duplicate rows")
        else:
            logger.success("✓ No duplicates found")

        return self

    def get_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame

        Returns:
            Processed DataFrame

        Raises:
            ValueError: If no data has been loaded
        """
        self._ensure_data_loaded()
        return self.df.copy()

    def save(self, filepath: Path) -> "DataProcessor":
        """
        Save processed data to CSV

        Args:
            filepath: Path where to save the CSV

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(filepath, index=False)
        logger.success(f"✓ Data saved to: {filepath}")

        return self

    def get_summary(self) -> Dict:
        """
        Get processing summary statistics

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None or self._original_shape is None:
            return {}

        return {
            "original_rows": self._original_shape[0],
            "original_cols": self._original_shape[1],
            "final_rows": self.df.shape[0],
            "final_cols": self.df.shape[1],
            "rows_removed": self._original_shape[0] - self.df.shape[0],
            "rows_removed_pct": f"{100 * (self._original_shape[0] - self.df.shape[0]) / self._original_shape[0]:.2f}%",
            "retention_rate": f"{100 * self.df.shape[0] / self._original_shape[0]:.2f}%",
        }

    def _ensure_data_loaded(self):
        """Internal helper to ensure data is loaded before operations"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
