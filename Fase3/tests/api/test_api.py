"""
Unit tests for FastAPI application.

Tests cover:
- Health check endpoint
- Model info endpoint
- Prediction endpoint
- Input validation
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
import joblib
from pathlib import Path
import json

from api.main import app
from api.schemas import PredictionInput, PredictionOutput


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint_success(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data

        # Status should be healthy if model loaded
        if data["model_loaded"]:
            assert data["status"] == "healthy"

    def test_model_info_endpoint(self, api_client):
        """Test model info endpoint returns metadata."""
        response = api_client.get("/model-info")

        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_type" in data


@pytest.mark.api
class TestPredictionEndpoint:
    """Test suite for prediction endpoint."""

    def test_predict_valid_input(self, api_client, sample_prediction_input):
        """Test prediction with valid input."""
        response = api_client.post("/predict", json=sample_prediction_input)

        # Should succeed if model is loaded
        if response.status_code == 200:
            data = response.json()

            # Check response structure
            assert "prediction" in data
            assert "probability" in data
            assert "risk_label" in data
            assert "model_version" in data

            # Check value types and ranges
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert data["risk_label"] in ["Bad Credit", "Good Credit"]

    def test_predict_invalid_input_missing_field(self, api_client):
        """Test prediction with missing required field."""
        invalid_input = {
            "age": 35,
            "amount": 5000,
            # Missing many required fields
        }

        response = api_client.post("/predict", json=invalid_input)

        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_age(self, api_client, sample_prediction_input):
        """Test prediction with invalid age value."""
        invalid_input = sample_prediction_input.copy()
        invalid_input["age"] = 150  # Invalid age

        response = api_client.post("/predict", json=invalid_input)

        # Should return validation error
        assert response.status_code == 422

    def test_predict_negative_amount(self, api_client, sample_prediction_input):
        """Test prediction with negative amount."""
        invalid_input = sample_prediction_input.copy()
        invalid_input["amount"] = -1000

        response = api_client.post("/predict", json=invalid_input)

        # Should return validation error
        assert response.status_code == 422

    def test_predict_out_of_range_checking_account(
        self, api_client, sample_prediction_input
    ):
        """Test prediction with out-of-range categorical value."""
        invalid_input = sample_prediction_input.copy()
        invalid_input["checking_account"] = 10  # Valid range is 1-4

        response = api_client.post("/predict", json=invalid_input)

        # Should return validation error
        assert response.status_code == 422


@pytest.mark.api
class TestInputValidation:
    """Test input validation schemas."""

    def test_prediction_input_schema_valid(self, sample_prediction_input):
        """Test PredictionInput schema with valid data."""
        # Should not raise any exception
        input_obj = PredictionInput(**sample_prediction_input)

        assert input_obj.age == 35
        assert input_obj.amount == 5000.0

    def test_prediction_input_schema_missing_field(self):
        """Test PredictionInput schema with missing field."""
        invalid_data = {"age": 35}  # Missing many required fields

        with pytest.raises(Exception):  # Pydantic ValidationError
            PredictionInput(**invalid_data)

    def test_prediction_input_schema_type_coercion(self):
        """Test PredictionInput schema type coercion."""
        data = {
            "age": "35",  # String instead of int
            "amount": "5000",  # String instead of float
            "checking_account": 1,
            "credit_history": 2,
            "dependents": 1,
            "duration": 24,
            "employment_duration": 3,
            "existing_credits": 1,
            "foreign_worker": 1,
            "housing": 2,
            "installment_rate": 2,
            "job": 2,
            "other_debtors": 1,
            "other_installment_plans": 1,
            "personal_status": 2,
            "property": 2,
            "purpose": 3,
            "residence_duration": 3,
            "savings_account": 1,
            "telephone": 1,
        }

        # Should coerce strings to numbers
        input_obj = PredictionInput(**data)
        assert isinstance(input_obj.age, int)
        assert isinstance(input_obj.amount, float)

    def test_prediction_output_schema(self):
        """Test PredictionOutput schema."""
        output_data = {
            "prediction": 1,
            "probability": 0.85,
            "risk_label": "Good Credit",
            "model_version": "1.0.0",
        }

        output_obj = PredictionOutput(**output_data)

        assert output_obj.prediction == 1
        assert output_obj.probability == 0.85
        assert output_obj.risk_label == "Good Credit"


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling."""

    def test_predict_with_malformed_json(self, api_client):
        """Test prediction with malformed JSON."""
        response = api_client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        # Should return 422 or 400
        assert response.status_code in [400, 422]

    def test_predict_with_empty_body(self, api_client):
        """Test prediction with empty request body."""
        response = api_client.post("/predict", json={})

        # Should return validation error
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, api_client):
        """Test accessing nonexistent endpoint."""
        response = api_client.get("/nonexistent")

        # Should return 404
        assert response.status_code == 404


@pytest.mark.api
@pytest.mark.slow
class TestAPIIntegration:
    """Integration tests for API with actual model."""

    def test_full_prediction_flow(self, api_client, sample_prediction_input):
        """Test complete prediction flow from request to response."""
        # 1. Check health
        health_response = api_client.get("/health")
        assert health_response.status_code == 200

        # 2. Get model info
        info_response = api_client.get("/model-info")

        # 3. Make prediction
        pred_response = api_client.post("/predict", json=sample_prediction_input)

        # If model is loaded, all should succeed
        if health_response.json()["model_loaded"]:
            assert info_response.status_code == 200
            assert pred_response.status_code == 200

            # Check prediction is consistent
            pred_data = pred_response.json()
            if pred_data["prediction"] == 1:
                assert pred_data["risk_label"] == "Good Credit"
                assert pred_data["probability"] > 0.5
            else:
                assert pred_data["risk_label"] == "Bad Credit"
                assert pred_data["probability"] <= 0.5

    def test_multiple_predictions_consistency(
        self, api_client, sample_prediction_input
    ):
        """Test that same input produces same prediction."""
        # Make multiple predictions with same input
        responses = []
        for _ in range(3):
            response = api_client.post("/predict", json=sample_prediction_input)
            if response.status_code == 200:
                responses.append(response.json())

        # If model is loaded, predictions should be consistent
        if len(responses) > 1:
            first_pred = responses[0]["prediction"]
            first_prob = responses[0]["probability"]

            for resp in responses[1:]:
                assert resp["prediction"] == first_pred
                assert abs(resp["probability"] - first_prob) < 0.001
