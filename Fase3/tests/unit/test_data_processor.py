"""
Unit tests for DataProcessor class.

Tests cover:
- Data loading
- Column translation
- Whitespace cleaning
- Type conversion
- Target validation
- Missing value handling
- Categorical validation
- Duplicate removal
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from fase3.core.data_processor import DataProcessor
from fase3.config import config


@pytest.mark.unit
class TestDataProcessor:
    """Test suite for DataProcessor class."""

    def test_initialization(self):
        """Test DataProcessor can be initialized."""
        processor = DataProcessor()
        assert processor.df is None
        assert processor.config is not None

    def test_load_raw_data_success(self, sample_csv_file):
        """Test successful data loading from CSV."""
        processor = DataProcessor()
        result = processor.load_raw_data(sample_csv_file)

        # Check method chaining
        assert result is processor

        # Check data loaded
        assert processor.df is not None
        assert len(processor.df) > 0
        assert processor._original_shape is not None

    def test_load_raw_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        processor = DataProcessor()

        with pytest.raises(FileNotFoundError):
            processor.load_raw_data(Path("nonexistent_file.csv"))

    def test_translate_columns(self, sample_csv_file):
        """Test column name translation from German to English."""
        # Create data with German column names
        df_german = pd.DataFrame(
            {"laufkont": [1, 2, 3], "laufzeit": [12, 24, 36], "kredit": [0, 1, 1]}
        )

        temp_path = sample_csv_file.parent / "german_data.csv"
        df_german.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(temp_path).translate_columns()

        # Check columns translated
        assert "checking_account" in processor.df.columns
        assert "duration" in processor.df.columns
        assert "credit_risk" in processor.df.columns

        # Check original German names removed
        assert "laufkont" not in processor.df.columns
        assert "laufzeit" not in processor.df.columns

    def test_clean_whitespace(self, sample_csv_file):
        """Test whitespace cleaning from string columns."""
        # Create data with whitespace
        df_whitespace = pd.DataFrame(
            {"col1": ["  value1  ", " value2 ", "value3"], "col2": [1, 2, 3]}
        )

        temp_path = sample_csv_file.parent / "whitespace_data.csv"
        df_whitespace.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(temp_path).clean_whitespace()

        # Check whitespace removed
        assert processor.df["col1"].iloc[0] == "value1"
        assert processor.df["col1"].iloc[1] == "value2"

    def test_convert_to_numeric(self, sample_raw_data, sample_csv_file):
        """Test conversion of columns to numeric types."""
        processor = DataProcessor()
        processor.load_raw_data(sample_csv_file).convert_to_numeric()

        # Check numeric columns
        numeric_cols = [c for c in processor.df.columns if c != "credit_risk"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(processor.df[col])

    def test_validate_target(self, sample_csv_file):
        """Test target variable validation."""
        processor = DataProcessor()
        processor.load_raw_data(sample_csv_file).validate_target()

        # Check target only has 0 and 1
        unique_values = processor.df["credit_risk"].unique()
        assert set(unique_values).issubset({0, 1, 0.0, 1.0})

    def test_validate_target_removes_invalid(self, sample_csv_file):
        """Test that invalid target values are removed."""
        # Create data with invalid target
        df_invalid = pd.DataFrame(
            {
                "checking_account": [1, 2, 3, 4],
                "credit_risk": [0, 1, 2, 0],  # 2 is invalid
            }
        )

        temp_path = sample_csv_file.parent / "invalid_target.csv"
        df_invalid.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(temp_path).convert_to_numeric().validate_target()

        # Check invalid row removed
        assert len(processor.df) == 3
        assert 2 not in processor.df["credit_risk"].values

    def test_handle_missing_values(self, sample_csv_file):
        """Test missing value imputation."""
        # Create data with missing values
        df_missing = pd.DataFrame(
            {
                "duration": [12, np.nan, 24, 36],
                "amount": [1000, 2000, np.nan, 3000],
                "credit_risk": [0, 1, 1, 0],
            }
        )

        temp_path = sample_csv_file.parent / "missing_data.csv"
        df_missing.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(temp_path).convert_to_numeric().handle_missing_values()

        # Check no missing values
        assert processor.df.isnull().sum().sum() == 0

    def test_remove_duplicates(self, sample_csv_file):
        """Test duplicate row removal."""
        # Create data with duplicates
        df_dupes = pd.DataFrame(
            {
                "col1": [1, 2, 2, 3],
                "col2": [10, 20, 20, 30],
                "credit_risk": [0, 1, 1, 0],
            }
        )

        temp_path = sample_csv_file.parent / "duplicate_data.csv"
        df_dupes.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(temp_path).remove_duplicates()

        # Check duplicates removed
        assert len(processor.df) == 3

    def test_get_data(self, sample_csv_file):
        """Test getting processed data."""
        processor = DataProcessor()
        processor.load_raw_data(sample_csv_file)

        df = processor.get_data()

        # Check returns DataFrame
        assert isinstance(df, pd.DataFrame)

        # Check is a copy (not reference)
        df.iloc[0, 0] = 999
        assert processor.df.iloc[0, 0] != 999

    def test_get_data_without_loading(self):
        """Test error when getting data before loading."""
        processor = DataProcessor()

        with pytest.raises(ValueError, match="No data loaded"):
            processor.get_data()

    def test_save(self, sample_csv_file, tmp_path):
        """Test saving processed data."""
        processor = DataProcessor()
        processor.load_raw_data(sample_csv_file)

        output_path = tmp_path / "output.csv"
        processor.save(output_path)

        # Check file created
        assert output_path.exists()

        # Check can be loaded
        df_saved = pd.read_csv(output_path)
        assert len(df_saved) == len(processor.df)

    def test_get_summary(self, sample_csv_file):
        """Test getting processing summary."""
        processor = DataProcessor()
        processor.load_raw_data(sample_csv_file).remove_duplicates()

        summary = processor.get_summary()

        # Check summary has expected keys
        assert "original_rows" in summary
        assert "final_rows" in summary
        assert "rows_removed" in summary
        assert "retention_rate" in summary

    def test_method_chaining(self, sample_csv_file):
        """Test that methods can be chained."""
        processor = DataProcessor()

        result = (
            processor.load_raw_data(sample_csv_file)
            .clean_whitespace()
            .convert_to_numeric()
            .validate_target()
            .handle_missing_values()
            .remove_duplicates()
        )

        # Check returns processor
        assert result is processor

        # Check data processed
        df = processor.get_data()
        assert df is not None
        assert len(df) > 0


@pytest.mark.unit
class TestDataProcessorValidation:
    """Test data validation logic."""

    def test_categorical_ranges_validation(self, sample_csv_file):
        """Test categorical variable range validation."""
        # Create data with out-of-range values
        df_invalid = pd.DataFrame(
            {
                "checking_account": [1, 2, 5, 3],  # 5 is invalid (max is 4)
                "credit_risk": [0, 1, 1, 0],
            }
        )

        temp_path = sample_csv_file.parent / "invalid_range.csv"
        df_invalid.to_csv(temp_path, index=False)

        processor = DataProcessor()
        processor.load_raw_data(
            temp_path
        ).convert_to_numeric().validate_categorical_ranges()

        # Check invalid row removed
        assert len(processor.df) == 3
        assert 5 not in processor.df["checking_account"].values
