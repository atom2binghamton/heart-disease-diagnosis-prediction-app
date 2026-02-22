# api/app.py
"""
FastAPI service for heart disease diagnosis prediction.
Loads the verified heart_pipeline and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import shared heart pipeline components so unpickling works
from heart_pipeline import (
    ColumnDropper,
    build_preprocessing,
    make_estimator_for_name,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("models/global_best_heart_optuna.pkl")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI service for predicting heart disease risk",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"  Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"✗ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"  Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Prediction request with list of instances (dicts of features).
    """
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "patient_id": "PT-4452",
                        "ca": 0,
                        "cp_id": 2,
                        "thal_id": 1,
                        "age": 52,
                        "sex": 1
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "predictions": [1],
                "probabilities": [88.5],
                "count": 1,
            }
        }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Could not convert to DataFrame: {e}",
        )

    # Required columns based on your simplified heart_pipeline
    required_columns = ["ca", "cp_id", "thal_id"]
    
    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        preds = model.predict(X)
        # Using predict_proba to provide the confidence percentage we discussed
        probs = model.predict_proba(X)[:, 1] 
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    preds_list = [int(p) for p in preds]
    probs_list = [round(float(p) * 100, 2) for p in probs]

    return PredictResponse(
        predictions=preds_list, 
        probabilities=probs_list, 
        count=len(preds_list)
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Heart Disease Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")