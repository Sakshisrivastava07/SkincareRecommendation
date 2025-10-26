# # skincareapi.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import List, Optional
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ---------- Config ----------
# MODEL_PATH = "xgb_model.pkl"
# MLB_PATH = "mlb.pkl"
# FEATURE_ORDER_PATH = "feature_order.pkl"  # saved during training
# DEFAULT_THRESHOLD = 0.3

# # ---------- Load model & artifacts ----------
# if not os.path.exists(MODEL_PATH) or not os.path.exists(MLB_PATH):
#     raise FileNotFoundError(
#         "xgb_model.pkl and mlb.pkl must exist in the working directory."
#     )

# model = joblib.load(MODEL_PATH)
# mlb = joblib.load(MLB_PATH)

# if os.path.exists(FEATURE_ORDER_PATH):
#     feature_order = joblib.load(FEATURE_ORDER_PATH)
# else:
#     try:
#         feature_order = list(model.feature_names_in_)
#     except Exception:
#         raise RuntimeError(
#             "Feature order not found. Save 'feature_order.pkl' during training."
#         )

# # ---------- FastAPI ----------
# app = FastAPI(
#     title="Skincare Recommender API",
#     description="Predicts multi-label skincare product recommendations",
#     version="1.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8000",
#         "http://127.0.0.1:8000",
#         "*",
#     ],  # allow frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------- Request Schema ----------
# class SkinRequest(BaseModel):
#     wrinkles_severity: float = Field(..., ge=0, le=10)
#     acne_severity: float = Field(..., ge=0, le=10)
#     dark_circle_severity: float = Field(..., ge=0, le=10)
#     pigmentation: float = Field(..., ge=0, le=10)
#     redness: float = Field(..., ge=0, le=10)
#     dark_spots: int = Field(..., ge=0, le=1)

#     # renamed vars, but accept JSON keys with underscores
#     oily: Optional[int] = Field(0, ge=0, le=1, alias="_oily")
#     dry: Optional[int] = Field(0, ge=0, le=1, alias="_dry")
#     normal: Optional[int] = Field(0, ge=0, le=1, alias="_normal")
#     sensitive: Optional[int] = Field(0, ge=0, le=1, alias="_sensitive")
#     fair: Optional[int] = Field(0, ge=0, le=1, alias="_fair")
#     dark: Optional[int] = Field(0, ge=0, le=1, alias="_dark")
#     olive: Optional[int] = Field(0, ge=0, le=1, alias="_olive")
#     brown: Optional[int] = Field(0, ge=0, le=1, alias="_brown")
#     medium: Optional[int] = Field(0, ge=0, le=1, alias="_medium")
#     combination: Optional[int] = Field(0, ge=0, le=1, alias="_combination")

#     class Config:
#         populate_by_name = True
#         allow_population_by_field_name = True
#         allow_population_by_alias = True


# # ---------- Response Schema ----------
# class RecommendationItem(BaseModel):
#     product: str
#     probability: float


# class PredictionResponse(BaseModel):
#     recommended: List[RecommendationItem]
#     probabilities: dict
#     threshold: float


# # ---------- Helper ----------
# def prepare_input_df(req: SkinRequest):
#     data = req.dict(by_alias=True)  # keep JSON keys with underscores
#     input_df = pd.DataFrame([data])

#     # add missing features as 0
#     for col in feature_order:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # keep only features that model expects
#     input_df = input_df[feature_order].astype(float)

#     # Debug prints
#     print("\n--- DEBUG ---")
#     print("Feature order (expected):", feature_order)
#     print("Input df columns:", input_df.columns.tolist())
#     print("Input df values:", input_df.values.tolist())
#     print("--------------\n")

#     return input_df


# # ---------- Endpoints ----------
# @app.post("/predict", response_model=PredictionResponse)
# def predict(request: SkinRequest, threshold: float = DEFAULT_THRESHOLD):
#     X = prepare_input_df(request)
#     try:
#         proba_list = model.predict_proba(X)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     pos_probs = np.array([arr[:, 1] for arr in proba_list]).T
#     binary_preds = (pos_probs >= threshold).astype(int)

#     recommended = mlb.inverse_transform(binary_preds)
#     recommended_labels = list(recommended[0])

#     # Build probability dictionary
#     prob_dict = {label: float(pos_probs[0, i]) for i, label in enumerate(mlb.classes_)}

#     # Attach probability with each recommendation
#     recommended_with_probs = [
#         {"product": label, "probability": prob_dict[label]}
#         for label in recommended_labels
#     ]

#     return PredictionResponse(
#         recommended=recommended_with_probs, probabilities=prob_dict, threshold=threshold
#     )


# @app.get("/")
# def root():
#     return {"status": "ok", "model_loaded": True, "n_labels": len(mlb.classes_)}


# skincareapi.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os

# ---------- Config ----------
MODEL_PATH = "xgb_model.pkl"
MLB_PATH = "mlb.pkl"
FEATURE_ORDER_PATH = "feature_order.pkl" 
DEFAULT_THRESHOLD = 0.3

# ---------- Load model & artifacts ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(MLB_PATH):
    raise FileNotFoundError(
        "xgb_model.pkl and mlb.pkl must exist in the working directory."
    )

try:
    model = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}")

# ---------- FastAPI ----------
app = FastAPI(
    title="Skincare Recommender API",
    description="Predicts multi-label skincare product recommendations",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
# Updated Request Schema
class SkinRequest(BaseModel):
    # Map frontend names to backend names using alias
    wrinkles: float = Field(..., ge=0, le=10, alias="wrinkles_severity")
    acne: float = Field(..., ge=0, le=10, alias="acne_severity")
    dark_circles: float = Field(..., ge=0, le=10, alias="dark_circle_severity")
    pigmentation: float = Field(..., ge=0, le=10)
    redness: float = Field(..., ge=0, le=10)
    dark_spots: int = Field(..., ge=0, le=1)

    # All checkboxes will now have aliases, and the Pydantic field names will be clean
    oily: Optional[int] = Field(0, ge=0, le=1, alias="_oily")
    dry: Optional[int] = Field(0, ge=0, le=1, alias="_dry")
    normal: Optional[int] = Field(0, ge=0, le=1, alias="_normal")
    sensitive: Optional[int] = Field(0, ge=0, le=1, alias="_sensitive")
    combination: Optional[int] = Field(0, ge=0, le=1, alias="_combination")
    fair: Optional[int] = Field(0, ge=0, le=1, alias="_fair")
    dark: Optional[int] = Field(0, ge=0, le=1, alias="_dark")
    olive: Optional[int] = Field(0, ge=0, le=1, alias="_olive")
    brown: Optional[int] = Field(0, ge=0, le=1, alias="_brown")
    medium: Optional[int] = Field(0, ge=0, le=1, alias="_medium")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True

# ---------- Response Schema ----------
class RecommendationItem(BaseModel):
    product: str
    probability: float

class PredictionResponse(BaseModel):
    recommended: List[RecommendationItem]
    probabilities: dict
    threshold: float

# ---------- Helper ----------
def prepare_input_df(req: SkinRequest):
    data = req.dict(by_alias=True)
    input_df = pd.DataFrame([data])
    
    # Ensure all features are present in the correct order
    final_df = pd.DataFrame(columns=feature_order)
    for col in feature_order:
        if col in input_df.columns:
            final_df[col] = input_df[col]
        else:
            final_df[col] = 0.0

    print("\n--- DEBUG ---")
    print("Feature order (expected):", feature_order)
    print("Final df columns:", final_df.columns.tolist())
    print("Final df values:", final_df.values.tolist())
    print("--------------\n")
    
    return final_df


# ---------- Endpoints ----------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: SkinRequest, threshold: float = DEFAULT_THRESHOLD):
    try:
        X = prepare_input_df(request)
        proba_list = model.predict_proba(X)
        pos_probs = np.array([arr[:, 1] for arr in proba_list]).T
        binary_preds = (pos_probs >= threshold).astype(int)

        recommended = mlb.inverse_transform(binary_preds)
        recommended_labels = list(recommended[0])

        prob_dict = {label: float(pos_probs[0, i]) for i, label in enumerate(mlb.classes_)}

        recommended_with_probs = [
            {"product": label, "probability": prob_dict[label]}
            for label in recommended_labels
        ]

        return PredictionResponse(
            recommended=recommended_with_probs, probabilities=prob_dict, threshold=threshold
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": True, "n_labels": len(mlb.classes_)}