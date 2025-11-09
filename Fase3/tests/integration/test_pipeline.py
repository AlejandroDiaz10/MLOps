import pytest
from pathlib import Path
from fase3.dataset import clean_data
from fase3.features import create_features
from fase3.modeling.train import train_model


def test_end_to_end_pipeline(tmp_path):
    """Test complete pipeline from raw data to trained model."""

    # Step 1: Clean data
    cleaned_path = clean_data(
        input_path=Path("data/raw/german_credit_modified.csv"),
        output_path=tmp_path / "cleaned.csv",
    )
    assert cleaned_path.exists()

    # Step 2: Feature engineering
    feature_paths = create_features(input_path=cleaned_path, output_dir=tmp_path)
    assert (tmp_path / "X_train.csv").exists()

    # Step 3: Train model
    model_path = train_model(model_name="logistic_regression", cv_folds=2)
    assert model_path.exists()
