"""
Feature engineering class with OOP principles
"""

from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from fase2.config import config


class FeatureEngineer:
    """
    Handles feature engineering operations including:
    - Outlier detection and removal
    - Train-test splitting
    - Feature scaling

    Uses method chaining for clean API.

    Attributes:
        config: Configuration object
        df: Current DataFrame
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        scaler: Fitted StandardScaler
    """

    def __init__(self, config_obj=None):
        """
        Initialize FeatureEngineer

        Args:
            config_obj: Optional custom configuration
        """
        self.config = config_obj or config
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[pd.DataFrame] = None
        self.X_test_scaled: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        logger.debug("FeatureEngineer initialized")

    def load_data(self, filepath: Optional[Path] = None) -> "FeatureEngineer":
        """
        Load cleaned data from CSV

        Args:
            filepath: Path to cleaned CSV. Uses default if None.

        Returns:
            self: For method chaining
        """
        if filepath is None:
            filepath = self.config.paths.interim_data_dir / "german_credit_cleaned.csv"

        try:
            logger.info(f"Loading cleaned data from: {filepath}")
            self.df = pd.read_csv(filepath)
            logger.success(f"✓ Data loaded: {self.df.shape}")
            return self
        except Exception as e:
            raise FileNotFoundError(f"Failed to load data: {str(e)}")

    def detect_outliers(
        self, columns: Optional[list] = None, method: str = "iqr"
    ) -> "FeatureEngineer":
        """
        Detect and remove outliers using IQR method

        Args:
            columns: List of columns to check. Uses continuous features if None.
            method: Outlier detection method ('iqr' only for now)

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()

        if columns is None:
            columns = [
                col
                for col in self.config.data.continuous_features
                if col in self.df.columns
            ]

        logger.info(
            f"Detecting outliers in {len(columns)} columns using {method.upper()} method..."
        )

        outlier_mask = pd.Series([False] * len(self.df), index=self.df.index)
        outlier_counts = {}

        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found, skipping...")
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_counts[col] = col_outliers.sum()
            outlier_mask = outlier_mask | col_outliers

        total_outliers = outlier_mask.sum()
        self.df = self.df[~outlier_mask].copy()

        # Log outlier details
        for col, count in outlier_counts.items():
            if count > 0:
                logger.debug(f"  [{col}] {count} outliers")

        logger.success(
            f"✓ Removed {total_outliers} outlier rows ({100*total_outliers/(total_outliers+len(self.df)):.2f}%)"
        )
        logger.info(f"  Dataset shape after outlier removal: {self.df.shape}")

        return self

    def split_target(self) -> "FeatureEngineer":
        """
        Separate features and target variable

        Returns:
            self: For method chaining
        """
        self._ensure_data_loaded()

        logger.info(
            f"Separating features and target '{self.config.data.target_col}'..."
        )

        self.X = self.df.drop(columns=[self.config.data.target_col])
        self.y = self.df[self.config.data.target_col]

        logger.success(f"✓ Features: {self.X.shape}, Target: {self.y.shape}")

        return self

    def train_test_split(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = True,
    ) -> "FeatureEngineer":
        """
        Split data into train and test sets

        Args:
            test_size: Proportion for test set. Uses config if None.
            random_state: Random seed. Uses config if None.
            stratify: Whether to stratify by target

        Returns:
            self: For method chaining
        """
        if self.X is None or self.y is None:
            raise ValueError("Must call split_target() before train_test_split()")

        test_size = test_size or self.config.model.test_size
        random_state = random_state or self.config.model.random_state

        logger.info(
            f"Splitting data (test_size={test_size}, random_state={random_state}, stratify={stratify})..."
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y if stratify else None,
        )

        logger.success(
            f"✓ Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples"
        )
        logger.info(
            f"  Train class distribution: {self.y_train.value_counts().to_dict()}"
        )
        logger.info(
            f"  Test class distribution: {self.y_test.value_counts().to_dict()}"
        )

        return self

    def scale_features(self, scaler_type: str = "standard") -> "FeatureEngineer":
        """
        Scale features using StandardScaler

        Args:
            scaler_type: Type of scaler ('standard' only for now)

        Returns:
            self: For method chaining
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Must call train_test_split() before scale_features()")

        logger.info(f"Scaling features using {scaler_type} scaler...")

        self.scaler = StandardScaler()

        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index,
        )

        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )

        logger.success("✓ Features scaled successfully")
        logger.debug(f"  Scaler mean (first 5): {self.scaler.mean_[:5]}")
        logger.debug(f"  Scaler scale (first 5): {self.scaler.scale_[:5]}")

        return self

    def get_train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get train-test split (scaled versions)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.X_train_scaled is None or self.X_test_scaled is None:
            raise ValueError("Must call scale_features() first")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def save_all(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save all processed data and artifacts

        Args:
            output_dir: Directory to save files. Uses config if None.

        Returns:
            Dictionary with paths to saved files
        """
        if output_dir is None:
            output_dir = self.config.paths.processed_data_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving processed datasets...")

        # Save datasets
        paths = {
            "X_train": output_dir / "X_train.csv",
            "X_test": output_dir / "X_test.csv",
            "y_train": output_dir / "y_train.csv",
            "y_test": output_dir / "y_test.csv",
            "scaler": self.config.paths.models_dir / "scaler.pkl",
        }

        self.X_train_scaled.to_csv(paths["X_train"], index=False)
        self.X_test_scaled.to_csv(paths["X_test"], index=False)
        self.y_train.to_csv(paths["y_train"], index=False, header=["credit_risk"])
        self.y_test.to_csv(paths["y_test"], index=False, header=["credit_risk"])

        # Save scaler
        self.config.paths.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, paths["scaler"])

        logger.success("✓ All files saved:")
        for name, path in paths.items():
            logger.info(f"  {name}: {path}")

        return paths

    def get_summary(self) -> Dict:
        """Get feature engineering summary"""
        summary = {}

        if self.df is not None:
            summary["total_samples"] = len(self.df)
            summary["total_features"] = self.df.shape[1] - 1  # -1 for target

        if self.X_train_scaled is not None:
            summary["train_samples"] = len(self.X_train_scaled)
            summary["test_samples"] = len(self.X_test_scaled)
            summary["n_features"] = self.X_train_scaled.shape[1]
            summary["test_size"] = (
                f"{100 * len(self.X_test_scaled) / (len(self.X_train_scaled) + len(self.X_test_scaled)):.1f}%"
            )

        return summary

    def _ensure_data_loaded(self):
        """Internal helper to ensure data is loaded"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
