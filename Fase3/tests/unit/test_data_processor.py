import pytest
import pandas as pd
from fase3.core.data_processor import DataProcessor


def test_translate_columns(sample_data):
    """Test column translation from German to English."""
    processor = DataProcessor()
    processor.df = sample_data
    result = processor.translate_columns()

    assert isinstance(result, DataProcessor)
    assert processor.df is not None


def test_clean_whitespace(sample_data):
    """Test whitespace cleaning."""
    processor = DataProcessor()
    processor.df = sample_data
    processor.df["test_col"] = "  value  "

    result = processor.clean_whitespace()
    assert processor.df["test_col"].iloc[0].strip() == "value"


def test_validate_target(sample_data):
    """Test target validation."""
    processor = DataProcessor()
    processor.df = sample_data

    result = processor.validate_target()
    assert processor.df["credit_risk"].isin([0.0, 1.0]).all()
