from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.churn_app_utils import (  # noqa: E402
    build_training_matrix,
    get_expected_columns,
    load_dataset,
    load_model,
    predict_local,
)


class CustomerPayload(BaseModel):
    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: str | None = None
    Dependents: str | None = None
    tenure: float | None = None
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float | None = None
    TotalCharges: float | None = None


app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="FastAPI service for telecom customer churn inference.",
)

MODEL, MODEL_PATH = load_model()
DATASET_DF, DATASET_PATH = load_dataset()
X_MATRIX, _, SCALER = build_training_matrix(DATASET_DF)
EXPECTED_COLUMNS = get_expected_columns(MODEL, X_MATRIX.columns)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "telco-churn-api",
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "dataset_path": str(DATASET_PATH),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: CustomerPayload) -> dict[str, Any]:
    result = predict_local(
        model=MODEL,
        scaler=SCALER,
        expected_columns=EXPECTED_COLUMNS,
        payload=payload.model_dump(exclude_none=True),
    )
    return {
        "prediction": result["prediction"],
        "prediction_label": result["prediction_label"],
        "churn_probability": result["churn_probability"],
    }

