"""
Pydantic schemas for API request/response validation.

These schemas ensure data validation and provide automatic API documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionInput(BaseModel):
    """
    Input schema for credit risk prediction.
    
    All features from the German Credit dataset.
    """
    
    # Account and credit history
    checking_account: int = Field(..., ge=1, le=4, description="Status of checking account")
    credit_history: int = Field(..., ge=0, le=4, description="Credit history status")
    savings_account: int = Field(..., ge=1, le=5, description="Savings account/bonds")
    
    # Credit details
    duration: int = Field(..., ge=1, le=72, description="Credit duration in months")
    amount: float = Field(..., ge=0, description="Credit amount")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate in % of disposable income")
    
    # Purpose
    purpose: int = Field(..., ge=0, le=10, description="Purpose of credit")
    
    # Employment and personal
    employment_duration: int = Field(..., ge=1, le=5, description="Present employment duration")
    personal_status: int = Field(..., ge=1, le=4, description="Personal status and sex")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    
    # Other debtors/guarantors
    other_debtors: int = Field(..., ge=1, le=3, description="Other debtors/guarantors")
    
    # Residence and property
    residence_duration: int = Field(..., ge=1, le=4, description="Present residence duration")
    property: int = Field(..., ge=1, le=4, description="Property type")
    
    # Other installment plans
    other_installment_plans: int = Field(..., ge=1, le=3, description="Other installment plans")
    
    # Housing
    housing: int = Field(..., ge=1, le=3, description="Housing type")
    
    # Job and dependents
    existing_credits: int = Field(..., ge=1, le=4, description="Number of existing credits at this bank")
    job: int = Field(..., ge=1, le=4, description="Job type")
    dependents: int = Field(..., ge=1, le=2, description="Number of people being liable to provide maintenance for")
    
    # Telephone and foreign worker
    telephone: int = Field(..., ge=1, le=2, description="Telephone registered under customer's name")
    foreign_worker: int = Field(..., ge=1, le=2, description="Foreign worker status")

    class Config:
        json_schema_extra = {
            "example": {
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
                "foreign_worker": 1,
            }
        }


class PredictionOutput(BaseModel):
    """
    Output schema for prediction response.
    """
    
    prediction: int = Field(..., description="Predicted class (0=Bad Credit, 1=Good Credit)")
    probability: float = Field(..., ge=0, le=1, description="Probability of Good Credit (class 1)")
    risk_label: str = Field(..., description="Human-readable risk label")
    model_version: str = Field(..., description="Model version used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "risk_label": "Good Credit",
                "model_version": "random_forest",
            }
        }
