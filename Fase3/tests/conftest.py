"""
Pytest configuration and shared fixtures.

This file provides reusable test fixtures for:
- Sample data generation
- Model loading
- API client setup
- Temporary file management
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Tuple, Dict

from fase3.config import config


# ============================================================================
# SESSION-SCOPED FIXTURES (created once per test session)
# ============================================================================


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test data."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    return temp_dir


@pytest.fixture(scope="session")
def sample_raw_data() -> pd.DataFrame:
    """
    Generate sample raw data mimicking German Credit dataset.

    Returns:
        DataFrame with 100 samples and all required features
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        # Categorical features
        "checking_account": np.random.randint(1, 5, n_samples),
        "credit_history": np.random.randint(0, 5, n_samples),
        "purpose": np.random.randint(0, 11, n_samples),
        "savings_account": np.random.randint(1, 6, n_samples),
        "employment_duration": np.random.randint(1, 6, n_samples),
        "personal_status": np.random.randint(1, 5, n_samples),
        "other_debtors": np.random.randint(1, 4, n_samples),
        "property": np.random.randint(1, 5, n_samples),
        "other_installment_plans": np.random.randint(1, 4, n_samples),
        "housing": np.random.randint(1, 4, n_samples),
        "job": np.random.randint(1, 5, n_samples),
        "dependents": np.random.randint(1, 3, n_samples),
        "telephone": np.random.randint(1, 3, n_samples),
        "foreign_worker": np.random.randint(1, 3, n_samples),
        # Continuous features
        "duration": np.random.randint(6, 72, n_samples),
        "amount": np.random.uniform(500, 15000, n_samples).round(2),
        "installment_rate": np.random.randint(1, 5, n_samples),
        "age": np.random.randint(19, 75, n_samples),
        "residence_duration": np.random.randint(1, 5, n_samples),
        "existing_credits": np.random.randint(1, 5, n_samples),
        # Target
        "credit_risk": np.random.randint(0, 2, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_clean_data(sample_raw_data) -> pd.DataFrame:
    """
    Generate cleaned version of sample data.

    Returns:
        Cleaned DataFrame (no NaNs, validated types)
    """
    df = sample_raw_data.copy()

    # Ensure no NaNs
    df = df.dropna()

    # Ensure correct dtypes
    for col in df.columns:
        if col != "amount":
            df[col] = df[col].astype(int)

    return df


# ============================================================================
# FUNCTION-SCOPED FIXTURES (created for each test function)
# ============================================================================


@pytest.fixture
def sample_train_test_split(
    sample_clean_data,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate train/test split from sample data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    df = sample_clean_data.copy()

    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_csv_file(sample_raw_data, tmp_path) -> Path:
    """
    Create temporary CSV file with sample data.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture

    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "test_data.csv"
    sample_raw_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_model_pipeline():
    """
    Create mock sklearn pipeline for testing.

    Returns:
        Fitted sklearn Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    # Create simple pipeline
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    return pipeline


@pytest.fixture
def sample_prediction_input() -> Dict:
    """
    Sample valid prediction input for API testing.

    Returns:
        Dictionary with all required fields
    """
    return {
        "checking_account": 1,
        "duration": 24,
        "credit_history": 2,
        "purpose": 3,
        "amount": 5000.0,
        "savings_account": 1,
        "employment_duration": 3,
        "installment_rate": 2,
        "personal_status": 2,
        "other_debtors": 1,
        "residence_duration": 3,
        "property": 2,
        "age": 35,
        "other_installment_plans": 1,
        "housing": 2,
        "existing_credits": 1,
        "job": 2,
        "dependents": 1,
        "telephone": 1,
        "foreign_worker": 1,
    }


@pytest.fixture
def api_client():
    """
    Create FastAPI test client.

    Returns:
        TestClient for API testing
    """
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create temporary directory structure for data."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    import random

    random.seed(42)


# ============================================================================
# MARKERS
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (slower)"
    )
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
