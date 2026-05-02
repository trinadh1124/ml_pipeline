"""
serve.py  —  FastAPI Inference Service
========================================
MODULE 2 KEY TOOL: Containerised inference endpoint.

Model is loaded from the MLflow registry — not from a local pickle.
When a new model version is promoted to production, restart the service
and it picks up the latest version automatically.

    uvicorn serve:app --reload --port 8000
    # Open http://localhost:8000/docs for the Swagger UI

DESIGN DECISIONS (from Module 2 design doc)
--------------------------------------------
  - Model loading from MLflow registry (not hardcoded path)
  - Input/output contracts via Pydantic
  - Structured logging (JSON, not print)
  - Health check endpoint for container orchestrators
  - Model version can be swapped without changing code
"""

import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("serve")

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a customer will churn. Model loaded from MLflow registry.",
    version="2.0.0",
)

# ── Load model and feature pipeline at startup ───────────────────────────────
MODEL = None
PIPELINE_INFO = None
MODEL_VERSION = "unknown"
PREDICTION_THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent


def _load_model():
    """Try MLflow registry first, fall back to local artefact."""
    global MODEL, PIPELINE_INFO, MODEL_VERSION, PREDICTION_THRESHOLD
    try:
        from src.registry.versioning import load_production_model, get_model_info
        from src.config.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        MODEL = load_production_model()
        info = get_model_info()
        prod = [v for v in info if v["stage"] == "production"]
        MODEL_VERSION = prod[-1]["version"] if prod else "unknown"
        logger.info(f"Loaded model from MLflow registry v{MODEL_VERSION}")
    except Exception as e:
        logger.warning(f"MLflow registry load failed: {e}. Falling back to local model.")
        import joblib
        model_dir = BASE_DIR / "models"
        model_files = sorted(model_dir.glob("*.pkl"))
        if model_files:
            MODEL = joblib.load(model_files[-1])
            MODEL_VERSION = model_files[-1].stem
        else:
            raise RuntimeError("No model found in registry or local models/ directory")

    # Load feature pipeline
    pipeline_path = BASE_DIR / "models" / "feature_pipeline.json"
    with open(pipeline_path) as f:
        PIPELINE_INFO = json.load(f)
    logger.info(f"Feature pipeline loaded ({len(PIPELINE_INFO['feature_cols'])} features)")

    best_params_path = BASE_DIR / "models" / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best_cfg = json.load(f)
        PREDICTION_THRESHOLD = float(best_cfg.get("threshold_cost", 0.50))
    logger.info(f"Prediction threshold set to {PREDICTION_THRESHOLD:.2f}")


@app.on_event("startup")
async def startup():
    _load_model()


# ── Pydantic I/O contracts ───────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """Input schema for a single customer prediction."""
    gender: str = Field(example="Male")
    senior_citizen: str = Field(example="No")
    partner: str = Field(example="Yes")
    dependents: str = Field(example="No")
    tenure_months: int = Field(ge=0, le=72, example=24)
    phone_service: str = Field(example="Yes")
    multiple_lines: str = Field(example="No")
    internet_service: str = Field(example="Fiber optic")
    online_security: str = Field(example="No")
    online_backup: str = Field(example="No")
    device_protection: str = Field(example="No")
    tech_support: str = Field(example="No")
    streaming_tv: str = Field(example="No")
    streaming_movies: str = Field(example="No")
    contract: str = Field(example="Month-to-month")
    paperless_billing: str = Field(example="Yes")
    payment_method: str = Field(example="Electronic check")
    monthly_charges: float = Field(ge=0, le=200, example=70.35)
    total_charges: float = Field(ge=0, le=10000, example=1397.47)


class PredictionOutput(BaseModel):
    """Output schema for a churn prediction."""
    churn_prediction: int
    churn_probability: float
    model_version: str
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check for container orchestrators."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": MODEL is not None,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    t0 = time.time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to DataFrame
    row = pd.DataFrame([customer.model_dump()])

    # Apply feature pipeline
    from src.features.build_features import build_features
    X, _, _ = build_features(
        row, fit=False,
        pipeline_path=str(BASE_DIR / "models" / "feature_pipeline.json"),
    )

    # Predict
    prob = float(MODEL.predict_proba(X)[0, 1])
    pred = int(prob >= PREDICTION_THRESHOLD)
    latency = (time.time() - t0) * 1000

    logger.info(f"prediction={pred} probability={prob:.3f} latency={latency:.1f}ms")

    return PredictionOutput(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        model_version=str(MODEL_VERSION),
        latency_ms=round(latency, 1),
    )


@app.post("/predict_batch")
def predict_batch(customers: list[CustomerInput]):
    """Predict churn for a batch of customers."""
    t0 = time.time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = pd.DataFrame([c.model_dump() for c in customers])
    from src.features.build_features import build_features
    X, _, _ = build_features(
        rows, fit=False,
        pipeline_path=str(BASE_DIR / "models" / "feature_pipeline.json"),
    )

    probs = MODEL.predict_proba(X)[:, 1].tolist()
    preds = [int(p >= PREDICTION_THRESHOLD) for p in probs]
    latency = (time.time() - t0) * 1000

    return {
        "predictions": [
            {"churn_prediction": int(p), "churn_probability": round(float(pr), 4)}
            for p, pr in zip(preds, probs)
        ],
        "model_version": str(MODEL_VERSION),
        "latency_ms": round(latency, 1),
        "batch_size": len(customers),
    }
