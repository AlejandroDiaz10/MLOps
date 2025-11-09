import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Generate sample German credit data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "duration": np.random.randint(6, 72, 100),
            "amount": np.random.randint(250, 18000, 100),
            "age": np.random.randint(19, 75, 100),
            "credit_risk": np.random.choice([0, 1], 100),
        }
    )


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 20), columns=[f"feature_{i}" for i in range(20)]
    )


@pytest.fixture
def sample_target():
    """Generate sample target vector."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100))
