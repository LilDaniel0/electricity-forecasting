# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
from datetime import datetime, timezone
import pandas as pd
import xgboost as xgb
import json
from pathlib import Path
from features_creation import make_features_for_next_step

ARTIFACTS = Path("artifacts")

# === cargar artefactos al iniciar ===
model = xgb.XGBRegressor()
model.load_model(ARTIFACTS / "xgb_model.json")
feature_order = json.load(open(ARTIFACTS / "feature_order.json"))
config = json.load(open(ARTIFACTS / "config.json"))
MIN_HIST = int(config.get("min_history_hours", 168))

app = FastAPI(title="AEP Forecasting API", version="0.1")


# ==== Schemas ====
class Observation(BaseModel):
    timestamp: datetime
    value: float

    @field_validator("timestamp")
    @classmethod
    def to_utc(cls, v: datetime):
        # normaliza a UTC sin tzinfo
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v


class PredictRequest(BaseModel):
    # Historial ordenado (del más viejo al más nuevo)
    history: List[Observation]


class PredictResponse(BaseModel):
    timestamp: datetime
    yhat: float


# ==== Endpoint ====
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.history) < MIN_HIST:
        raise HTTPException(
            status_code=400,
            detail=f"Se requieren al menos {MIN_HIST} horas de historial.",
        )

    # convertir a Serie pandas
    hist_df = pd.DataFrame(
        [{"Datetime": o.timestamp, "AEP_MW": o.value} for o in req.history]
    )
    hist_df = hist_df.sort_values("Datetime").set_index("Datetime")
    # next timestamp = última hora + 1h
    next_when = hist_df.index.max() + pd.Timedelta(hours=1)

    X = make_features_for_next_step(hist_df["AEP_MW"], when=next_when)

    # reordenar columnas como en entrenamiento
    X = X.reindex(columns=feature_order)

    if X.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="Historial insuficiente para construir todas las features (lags/rollings).",
        )

    yhat = float(model.predict(X)[0])
    return PredictResponse(timestamp=next_when, yhat=yhat)
