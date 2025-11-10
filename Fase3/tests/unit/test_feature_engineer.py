"""
Unit tests for FeatureEngineer class.

Tests cover:
- Data loading
- Outlier detection
- Train-test split
- Feature scaling
- Data saving
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from fase3.core.feature_engineer import FeatureEngineer


@pytest.mark.unit
class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_initialization(self):
        """Test FeatureEngineer can be initialized."""
        engineer = FeatureEngineer()
        assert engineer.df is None
        assert engineer.X is None
        assert engineer.y is None

    def test_load_data(self, sample_clean_data, tmp_path):
        """Test loading cleaned data."""
        # Save sample data
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        result = engineer.load_data(data_path)

        # Check method chaining
        assert result is engineer

        # Check data loaded
        assert engineer.df is not None
        assert len(engineer.df) == len(sample_clean_data)

    def test_load_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        engineer = FeatureEngineer()

        with pytest.raises(FileNotFoundError):
            engineer.load_data(Path("nonexistent.csv"))

    def test_detect_outliers(self, sample_clean_data, tmp_path):
        """Test outlier detection and removal."""
        # Add some extreme outliers
        df = sample_clean_data.copy()
        df.loc[0, "amount"] = 999999  # Extreme outlier

        data_path = tmp_path / "data_with_outliers.csv"
        df.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).detect_outliers()

        # Check outliers removed
        assert len(engineer.df) < len(df)
        assert 999999 not in engineer.df["amount"].values

    def test_split_target(self, sample_clean_data, tmp_path):
        """Test separating features and target."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target()

        # Check X and y created
        assert engineer.X is not None
        assert engineer.y is not None

        # Check target not in X
        assert "credit_risk" not in engineer.X.columns

        # Check shapes match
        assert len(engineer.X) == len(engineer.y)

    def test_train_test_split(self, sample_clean_data, tmp_path):
        """Test train-test splitting."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split(
            test_size=0.2, random_state=42
        )

        # Check splits created
        assert engineer.X_train is not None
        assert engineer.X_test is not None
        assert engineer.y_train is not None
        assert engineer.y_test is not None

        # Check sizes
        total_samples = len(sample_clean_data)
        assert len(engineer.X_train) == int(total_samples * 0.8)
        assert len(engineer.X_test) == int(total_samples * 0.2)

    def test_train_test_split_stratified(self, sample_clean_data, tmp_path):
        """Test stratified train-test split."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split(stratify=True)

        # Check class distribution preserved
        train_ratio = (engineer.y_train == 1).sum() / len(engineer.y_train)
        test_ratio = (engineer.y_test == 1).sum() / len(engineer.y_test)

        # Should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1

    def test_scale_features(self, sample_clean_data, tmp_path):
        """Test feature scaling."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        # Check scaled data created
        assert engineer.X_train_scaled is not None
        assert engineer.X_test_scaled is not None

        # Check scaler fitted
        assert engineer.scaler is not None

        # Check scaled values have mean ~0 and std ~1
        mean_train = engineer.X_train_scaled.mean().mean()
        std_train = engineer.X_train_scaled.std().mean()

        assert abs(mean_train) < 0.1  # Close to 0
        assert abs(std_train - 1.0) < 0.1  # Close to 1

    def test_scale_features_preserves_shape(self, sample_clean_data, tmp_path):
        """Test that scaling preserves DataFrame shape and columns."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        # Check shapes preserved
        assert engineer.X_train_scaled.shape == engineer.X_train.shape
        assert engineer.X_test_scaled.shape == engineer.X_test.shape

        # Check columns preserved
        assert list(engineer.X_train_scaled.columns) == list(engineer.X_train.columns)

    def test_get_train_test_split(self, sample_clean_data, tmp_path):
        """Test getting train-test split."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        X_train, X_test, y_train, y_test = engineer.get_train_test_split()

        # Check returns correct types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Check returns scaled versions
        assert X_train.equals(engineer.X_train_scaled)

    def test_save_all(self, sample_clean_data, tmp_path):
        """Test saving all processed datasets."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        output_dir = tmp_path / "output"

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        paths = engineer.save_all(output_dir)

        # Check all files saved
        assert paths["X_train"].exists()
        assert paths["X_test"].exists()
        assert paths["y_train"].exists()
        assert paths["y_test"].exists()
        assert paths["scaler"].exists()

        # Check files can be loaded
        X_train_loaded = pd.read_csv(paths["X_train"])
        assert len(X_train_loaded) == len(engineer.X_train_scaled)

    def test_get_summary(self, sample_clean_data, tmp_path):
        """Test getting feature engineering summary."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        summary = engineer.get_summary()

        # Check summary keys
        assert "total_samples" in summary
        assert "train_samples" in summary
        assert "test_samples" in summary
        assert "n_features" in summary

    def test_method_chaining(self, sample_clean_data, tmp_path):
        """Test complete pipeline with method chaining."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()

        result = (
            engineer.load_data(data_path)
            .detect_outliers()
            .split_target()
            .train_test_split()
            .scale_features()
        )

        # Check returns engineer
        assert result is engineer

        # Check data processed
        X_train, X_test, y_train, y_test = engineer.get_train_test_split()
        assert X_train is not None
        assert len(X_train) > 0


@pytest.mark.unit
class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""

    def test_scale_before_split_raises_error(self, sample_clean_data, tmp_path):
        """Test error when trying to scale before splitting."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target()

        with pytest.raises(ValueError, match="train_test_split"):
            engineer.scale_features()

    def test_split_before_load_raises_error(self):
        """Test error when trying to split before loading."""
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="No data loaded"):
            engineer.split_target()

    def test_outlier_detection_with_no_outliers(self, sample_clean_data, tmp_path):
        """Test outlier detection when no outliers present."""
        data_path = tmp_path / "clean_data.csv"
        sample_clean_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path)

        original_len = len(engineer.df)
        engineer.detect_outliers()

        # Should keep most data (outliers are rare in clean data)
        assert len(engineer.df) >= original_len * 0.9
