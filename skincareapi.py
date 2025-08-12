# skincareapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Config ----------
MODEL_PATH = "xgb_model.pkl"
MLB_PATH = "mlb.pkl"
FEATURE_ORDER_PATH = "feature_order.pkl"  # saved during training
DEFAULT_THRESHOLD = 0.3

# ---------- Load model & artifacts ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(MLB_PATH):
    raise FileNotFoundError("xgb_model.pkl and mlb.pkl must exist in the working directory.")

model = joblib.load(MODEL_PATH)
mlb = joblib.load(MLB_PATH)

if os.path.exists(FEATURE_ORDER_PATH):
    feature_order = joblib.load(FEATURE_ORDER_PATH)
else:
    try:
        feature_order = list(model.feature_names_in_)
    except Exception:
        raise RuntimeError("Feature order not found. Save 'feature_order.pkl' during training.")

# ---------- FastAPI ----------
app = FastAPI(
    title="Skincare Recommender API",
    description="Predicts multi-label skincare product recommendations",
    version="1.0"
)

class SkinRequest(BaseModel):
    wrinkles_severity: float = Field(..., ge=0, le=10)
    acne_severity: float = Field(..., ge=0, le=10)
    dark_circle_severity: float = Field(..., ge=0, le=10)
    pigmentation: float = Field(..., ge=0, le=10)
    redness: float = Field(..., ge=0, le=10)
    dark_spots: int = Field(..., ge=0, le=1)

    # renamed vars, but accept JSON keys with underscores
    combination: Optional[int] = Field(0, ge=0, le=1, alias="_combination")
    dry: Optional[int] = Field(0, ge=0, le=1, alias="_dry")
    normal: Optional[int] = Field(0, ge=0, le=1, alias="_normal")
    oily: Optional[int] = Field(0, ge=0, le=1, alias="_oily")
    sensitive: Optional[int] = Field(0, ge=0, le=1, alias="_sensitive")
    brown: Optional[int] = Field(0, ge=0, le=1, alias="_brown")
    dark: Optional[int] = Field(0, ge=0, le=1, alias="_dark")
    fair: Optional[int] = Field(0, ge=0, le=1, alias="_fair")
    medium: Optional[int] = Field(0, ge=0, le=1, alias="_medium")
    olive: Optional[int] = Field(0, ge=0, le=1, alias="_olive")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    recommended: List[str]
    probabilities: dict
    threshold: float

def prepare_input_df(req: SkinRequest):
    data = req.dict(by_alias=True)  # keep JSON keys with underscores
    input_df = pd.DataFrame([data])

    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order].astype(float)
    return input_df

@app.post("/predict", response_model=PredictionResponse)
def predict(request: SkinRequest, threshold: float = DEFAULT_THRESHOLD):
    X = prepare_input_df(request)
    try:
        proba_list = model.predict_proba(X)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pos_probs = np.array([arr[:, 1] for arr in proba_list]).T
    binary_preds = (pos_probs >= threshold).astype(int)

    recommended = mlb.inverse_transform(binary_preds)
    recommended_list = list(recommended[0])
    prob_dict = {label: float(pos_probs[0, i]) for i, label in enumerate(mlb.classes_)}

    return PredictionResponse(
        recommended=recommended_list,
        probabilities=prob_dict,
        threshold=threshold
    )

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": True, "n_labels": len(mlb.classes_)}
