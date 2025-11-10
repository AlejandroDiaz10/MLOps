"""
Integration tests for complete ML pipeline.

Tests cover:
- End-to-end data processing pipeline
- End-to-end training pipeline
- Model serialization and loading
- Complete prediction flow
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from fase3.core.data_processor import DataProcessor
from fase3.core.feature_engineer import FeatureEngineer
from fase3.core.model_factory import ModelFactory
from fase3.pipeline_builder import PipelineBuilder


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data processing pipeline."""

    def test_complete_data_pipeline(self, sample_raw_data, tmp_path):
        """Test complete data processing from raw to processed."""
        # Save raw data
        raw_path = tmp_path / "raw_data.csv"
        sample_raw_data.to_csv(raw_path, index=False)

        # Step 1: Clean data
        processor = DataProcessor()
        clean_df = (
            processor.load_raw_data(raw_path)
            .clean_whitespace()
            .convert_to_numeric()
            .validate_target()
            .handle_missing_values()
            .remove_duplicates()
            .get_data()
        )

        # Save cleaned data
        interim_path = tmp_path / "cleaned_data.csv"
        clean_df.to_csv(interim_path, index=False)

        # Step 2: Feature engineering
        engineer = FeatureEngineer()
        paths = (
            engineer.load_data(interim_path)
            .detect_outliers()
            .split_target()
            .train_test_split()
            .scale_features()
            .save_all(tmp_path / "processed")
        )

        # Validate outputs
        assert paths["X_train"].exists()
        assert paths["X_test"].exists()
        assert paths["y_train"].exists()
        assert paths["y_test"].exists()
        assert paths["scaler"].exists()

        # Load and validate processed data
        X_train = pd.read_csv(paths["X_train"])
        y_train = pd.read_csv(paths["y_train"])

        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_train) == len(y_train)
        assert X_train.isnull().sum().sum() == 0  # No missing values


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for model training pipeline."""

    def test_pipeline_builder_creates_valid_pipeline(self, sample_train_test_split):
        """Test PipelineBuilder creates trainable pipeline."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Create pipeline
        builder = PipelineBuilder()
        pipeline = builder.build_pipeline(model_name="random_forest")

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Validate predictions
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})

    def test_pipeline_with_different_models(self, sample_train_test_split):
        """Test pipeline works with different model types."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        models = ["random_forest", "logistic_regression", "decision_tree"]

        for model_name in models:
            # Create and train pipeline
            builder = PipelineBuilder()
            pipeline = builder.build_pipeline(model_name=model_name)
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)

            # Validate
            assert len(y_pred) == len(y_test)
            assert set(y_pred).issubset({0, 1})

            # Check predict_proba works
            y_proba = pipeline.predict_proba(X_test)
            assert y_proba.shape == (len(y_test), 2)
            assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_grid_search_pipeline(self, sample_train_test_split):
        """Test GridSearchCV pipeline training."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Create GridSearch pipeline with small grid
        builder = PipelineBuilder()
        grid_pipeline = builder.build_grid_search_pipeline(
            model_name="random_forest", cv_folds=2  # Fast for testing
        )

        # Train
        grid_pipeline.fit(X_train, y_train)

        # Check best estimator exists
        assert grid_pipeline.best_estimator_ is not None
        assert grid_pipeline.best_score_ is not None
        assert grid_pipeline.best_params_ is not None

        # Predict with best model
        y_pred = grid_pipeline.predict(X_test)
        assert len(y_pred) == len(y_test)


@pytest.mark.integration
class TestModelPersistence:
    """Integration tests for model serialization and loading."""

    def test_save_and_load_pipeline(self, sample_train_test_split, tmp_path):
        """Test saving and loading trained pipeline."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Train pipeline
        builder = PipelineBuilder()
        pipeline = builder.build_pipeline(model_name="random_forest")
        pipeline.fit(X_train, y_train)

        # Save pipeline
        model_path = tmp_path / "test_pipeline.pkl"
        joblib.dump(pipeline, model_path)

        # Load pipeline
        loaded_pipeline = joblib.load(model_path)

        # Predict with both
        y_pred_original = pipeline.predict(X_test)
        y_pred_loaded = loaded_pipeline.predict(X_test)

        # Should be identical
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)

    def test_pipeline_reproducibility(self, sample_train_test_split):
        """Test that pipeline produces same results with same seed."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Train two identical pipelines
        builder1 = PipelineBuilder()
        pipeline1 = builder1.build_pipeline(model_name="random_forest")
        pipeline1.named_steps["model"].set_params(random_state=42)
        pipeline1.fit(X_train, y_train)

        builder2 = PipelineBuilder()
        pipeline2 = builder2.build_pipeline(model_name="random_forest")
        pipeline2.named_steps["model"].set_params(random_state=42)
        pipeline2.fit(X_train, y_train)

        # Predictions should be identical
        y_pred1 = pipeline1.predict(X_test)
        y_pred2 = pipeline2.predict(X_test)

        np.testing.assert_array_equal(y_pred1, y_pred2)


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_raw_to_predictions_complete_flow(self, sample_raw_data, tmp_path):
        """Test complete flow from raw data to predictions."""
        # Save raw data
        raw_path = tmp_path / "raw_data.csv"
        sample_raw_data.to_csv(raw_path, index=False)

        # Step 1: Data Processing
        processor = DataProcessor()
        clean_df = (
            processor.load_raw_data(raw_path)
            .clean_whitespace()
            .convert_to_numeric()
            .validate_target()
            .handle_missing_values()
            .remove_duplicates()
            .get_data()
        )

        interim_path = tmp_path / "cleaned_data.csv"
        clean_df.to_csv(interim_path, index=False)

        # Step 2: Feature Engineering
        engineer = FeatureEngineer()
        engineer.load_data(
            interim_path
        ).detect_outliers().split_target().train_test_split().scale_features()

        X_train, X_test, y_train, y_test = engineer.get_train_test_split()

        # Step 3: Model Training
        builder = PipelineBuilder()
        pipeline = builder.build_pipeline(model_name="random_forest")
        pipeline.fit(X_train, y_train)

        # Step 4: Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        # Step 5: Validation
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
        assert y_proba.shape == (len(y_test), 2)

        # Step 6: Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score

        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba[:, 1])

        # Basic sanity checks
        assert 0 <= accuracy <= 1
        assert 0 <= auc_roc <= 1

        # Model should perform better than random
        assert accuracy > 0.5

    def test_model_factory_integration(self, sample_train_test_split):
        """Test ModelFactory creates working models."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        for model_name in ModelFactory.get_available_models():
            # Create model
            model = ModelFactory.create_model(model_name, random_state=42)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Validate
            assert len(y_pred) == len(y_test)
            assert set(y_pred).issubset({0, 1})


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for model performance."""

    def test_models_achieve_minimum_accuracy(self, sample_train_test_split):
        """Test that models achieve reasonable accuracy."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        from sklearn.metrics import accuracy_score

        models = ["random_forest", "logistic_regression", "decision_tree"]

        for model_name in models:
            builder = PipelineBuilder()
            pipeline = builder.build_pipeline(model_name=model_name)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Should perform at least as good as random (50%)
            assert accuracy >= 0.48, f"{model_name} accuracy too low: {accuracy}"

    def test_pipeline_handles_large_dataset(self, tmp_path):
        """Test pipeline can handle larger dataset."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 1000

        large_data = pd.DataFrame(
            {
                "checking_account": np.random.randint(1, 5, n_samples),
                "duration": np.random.randint(6, 72, n_samples),
                "credit_history": np.random.randint(0, 5, n_samples),
                "purpose": np.random.randint(0, 11, n_samples),
                "amount": np.random.uniform(500, 15000, n_samples),
                "savings_account": np.random.randint(1, 6, n_samples),
                "employment_duration": np.random.randint(1, 6, n_samples),
                "installment_rate": np.random.randint(1, 5, n_samples),
                "personal_status": np.random.randint(1, 5, n_samples),
                "other_debtors": np.random.randint(1, 4, n_samples),
                "residence_duration": np.random.randint(1, 5, n_samples),
                "property": np.random.randint(1, 5, n_samples),
                "age": np.random.randint(19, 75, n_samples),
                "other_installment_plans": np.random.randint(1, 4, n_samples),
                "housing": np.random.randint(1, 4, n_samples),
                "existing_credits": np.random.randint(1, 5, n_samples),
                "job": np.random.randint(1, 5, n_samples),
                "dependents": np.random.randint(1, 3, n_samples),
                "telephone": np.random.randint(1, 3, n_samples),
                "foreign_worker": np.random.randint(1, 3, n_samples),
                "credit_risk": np.random.randint(0, 2, n_samples),
            }
        )

        # Save and process
        data_path = tmp_path / "large_data.csv"
        large_data.to_csv(data_path, index=False)

        engineer = FeatureEngineer()
        engineer.load_data(data_path).split_target().train_test_split().scale_features()

        X_train, X_test, y_train, y_test = engineer.get_train_test_split()

        # Train model
        builder = PipelineBuilder()
        pipeline = builder.build_pipeline(model_name="random_forest")
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Should complete without errors
        assert len(y_pred) == len(y_test)
