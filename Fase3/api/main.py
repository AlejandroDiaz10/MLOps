"""
FastAPI service for credit risk prediction using the best trained model.

This API:
- Loads the best model selected based on test AUC-ROC
- Provides health check endpoint
- Provides prediction endpoint with validation
- Uses Pydantic for request/response validation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pathlib import Path
from loguru import logger
from contextlib import asynccontextmanager

from fase3.config import config
from api.schemas import PredictionInput, PredictionOutput


# Load model from local file
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown event handler.
    Loads the best model on startup.
    """
    # --- Startup ---
    logger.info("🚀 Starting API server...")

    # Try to load best_model first, fallback to specific model
    model_path = config.paths.models_dir / "best_model_pipeline.pkl"

    if not model_path.exists():
        logger.warning(f"⚠️ Best model not found at {model_path}")
        # Fallback to random_forest
        model_path = config.paths.models_dir / "random_forest_pipeline.pkl"
        logger.info(f"   Using fallback model: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found. Please train models first:\n"
            f"  python -m fase3.modeling.train --model-name random_forest\n"
            f"  python -m fase3.modeling.select_best_model"
        )

    try:
        app.state.model = joblib.load(model_path)
        app.state.model_path = str(model_path)
        logger.success(f"✅ Model loaded from: {model_path}")

        # Load metadata if available
        metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                app.state.model_metadata = json.load(f)
            logger.info(f"✅ Model metadata loaded")
        else:
            app.state.model_metadata = {}

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    yield

    # --- Shutdown ---
    logger.info("🛑 Shutting down API server...")


app = FastAPI(
    title="German Credit Risk Prediction API",
    description="API for credit risk prediction using best trained model",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "German Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
        },
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns API status and model loading status.
    """
    model_loaded = hasattr(app.state, "model") and app.state.model is not None

    response = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }

    if model_loaded:
        response["model_path"] = app.state.model_path

    return response


@app.get("/model-info")
def model_info():
    """
    Get information about the loaded model.

    Returns model metadata including training metrics and parameters.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = getattr(app.state, "model_metadata", {})

    return {
        "model_path": app.state.model_path,
        "model_name": metadata.get("model_name", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "training_date": metadata.get("training_date", "unknown"),
        "test_metrics": metadata.get("test_metrics", {}),
        "best_params": metadata.get("grid_search", {}).get("best_params", {}),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Make credit risk prediction.

    Args:
        input_data: Credit application data

    Returns:
        Prediction result with probability and risk label

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])

        # Make prediction
        prediction = int(app.state.model.predict(df)[0])

        # Get probability if available
        if hasattr(app.state.model, "predict_proba"):
            probability = float(app.state.model.predict_proba(df)[0][1])
        else:
            probability = float(prediction)

        # Get risk label
        risk_label = "Good Credit" if prediction == 1 else "Bad Credit"

        # Get model version from metadata
        metadata = getattr(app.state, "model_metadata", {})
        model_version = metadata.get("model_name", "1.0.0")

        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            risk_label=risk_label,
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Example usage documentation
"""
USAGE EXAMPLES:

1. Start the server:
   uvicorn api.main:app --reload --port 8000

2. Health check:
   curl http://localhost:8000/health

3. Model info:
   curl http://localhost:8000/model-info

4. Make prediction:
   curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{
       "checking_account": 1,
       "duration": 24,
       "credit_history": 2,
       "purpose": 3,
       "amount": 5000,
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
       "foreign_worker": 1
     }'

5. Interactive API docs:
   Open browser: http://localhost:8000/docs
"""
