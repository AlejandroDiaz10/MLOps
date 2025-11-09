from pydantic import BaseModel, Field, validator
from typing import List


class PredictionInput(BaseModel):
    """Input schema for credit risk prediction."""

    duration: int = Field(..., ge=1, le=72, description="Credit duration in months")
    amount: float = Field(..., ge=0, description="Credit amount")
    installment_rate: int = Field(..., ge=1, le=4)
    age: int = Field(..., ge=18, le=100)
    # ... resto de features

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 24,
                "amount": 5000,
                "age": 35,
                # ...
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction."""

    prediction: int = Field(..., description="0=Bad Credit, 1=Good Credit")
    probability: float = Field(..., ge=0, le=1)
    model_version: str
