from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
from fase3.config import config
from api.schemas import PredictionInput, PredictionOutput
from contextlib import asynccontextmanager


# Load model from MLflow
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    model_uri = f"models:/{config.api.model_name}/{config.api.model_stage}"
    app.state.model = mlflow.sklearn.load_model(model_uri)
    print("✅ MLflow model loaded on startup")
    yield

    # --- Shutdown (optional) ---
    # If any cleanup is needed, it can be done here
    print("🛑 Application shutdown")


app = FastAPI(
    title="German Credit Risk API",
    description="API for credit risk prediction using MLflow model",
    version="1.0.0",
    lifespan=lifespan,
)


# Optional: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Make credit risk prediction."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data.dict()])

        # Predict
        prediction = int(app.state.model.predict(df)[0])
        probability = float(app.state.model.predict_proba(df)[0][1])

        return PredictionOutput(
            prediction=prediction, probability=probability, model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
uvicorn api.main:app --reload --port 8000

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 24, "amount": 5000, "age": 35, ...}'
"""
