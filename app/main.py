import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        Path(__file__).resolve().parent.parent / "models" / "model.joblib",
    )
)

app = FastAPI(title="MP108 Pharmacy Daily Sales API", version="1.0")

_model = None


def get_model():
    """Load the model lazily so the API (and tests) can start without it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model file not found")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    model = get_model()
    # The saved model is a full sklearn Pipeline (scaler + one-hot inside),
    # so raw columns go straight in.
    df = pd.DataFrame([payload.model_dump()])
    try:
        prediction = float(model.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return PredictionResponse(predicted_units_sold=prediction)