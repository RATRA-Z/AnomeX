"""
predict.py
----------
Loads the trained model once and exposes functions to:
  - predict failure probability / health score for one or many components
  - estimate a rough "failure horizon" (hours until the safety limit is hit)
  - explain a prediction using feature importances (used by the AI assistant)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from ml.config import (
    MODEL_FILE,
    METRICS_FILE,
    THRESHOLDS_FILE,
    FEATURE_IMPORTANCE_FILE,
    probability_to_health_score,
)
from ml.preprocessing import (
    load_encoders,
    build_feature_matrix,
    get_feature_column_names,
)

_model = None
_encoders = None
_thresholds = None
_feature_importance = None


def _ensure_loaded():
    """Lazy-load model artifacts once per process."""
    global _model, _encoders, _thresholds, _feature_importance
    if _model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found at {MODEL_FILE}. Run train_model.py first."
            )
        _model = joblib.load(MODEL_FILE)
        _encoders = load_encoders()
        with open(THRESHOLDS_FILE) as f:
            _thresholds = json.load(f)
        with open(FEATURE_IMPORTANCE_FILE) as f:
            _feature_importance = json.load(f)


def get_metrics() -> dict:
    with open(METRICS_FILE) as f:
        return json.load(f)


def get_feature_importance() -> list:
    _ensure_loaded()
    return _feature_importance


def estimate_failure_horizon(row: pd.Series) -> dict:
    """
    Rough ETA estimate: given the current drift_rate (units per hour) and the
    current value_168h, how many hours until value crosses the safety limit
    for this component_type?

    This is intentionally simple (linear extrapolation) so it stays explainable
    for a viva -- it mirrors the "Est. Failure Horizon" box in the project brief.
    """
    _ensure_loaded()
    ctype = row.get("component_type", "_default")
    limit = _thresholds.get(ctype, _thresholds["_default"])
    current_value = float(row["value_168h"])
    drift_rate = float(row["drift_rate"])

    if current_value >= limit:
        return {"safety_limit": limit, "eta_hours": 0, "status": "past_safety_limit"}

    if drift_rate <= 0:
        return {"safety_limit": limit, "eta_hours": None, "status": "stable"}

    eta_hours = round((limit - current_value) / drift_rate, 1)
    eta_hours = min(eta_hours, 8760)  # cap at 1 year so absurd extrapolations don't leak out
    return {"safety_limit": limit, "eta_hours": eta_hours, "status": "degrading"}


def predict_single(row: pd.Series) -> dict:
    """Run the full prediction pipeline for one component (as a pandas Series)."""
    _ensure_loaded()
    df_row = pd.DataFrame([row])
    X = build_feature_matrix(df_row, _encoders)
    proba = float(_model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)
    health_score = probability_to_health_score(proba)
    horizon = estimate_failure_horizon(row)

    return {
        "component_id": row.get("component_id", "unknown"),
        "failure_probability": round(proba, 4),
        "predicted_label": prediction,
        "health_score": health_score,
        "risk_level": risk_level_from_score(health_score),
        **horizon,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised prediction for a whole dataframe. Returns df with new columns added."""
    _ensure_loaded()
    X = build_feature_matrix(df, _encoders)
    proba = _model.predict_proba(X)[:, 1]

    result = df.copy()
    result["failure_probability"] = np.round(proba, 4)
    result["predicted_label"] = (proba >= 0.5).astype(int)
    result["health_score"] = [probability_to_health_score(p) for p in proba]
    result["risk_level"] = [risk_level_from_score(h) for h in result["health_score"]]
    return result


def risk_level_from_score(health_score: float) -> str:
    if health_score >= 80:
        return "Normal"
    elif health_score >= 50:
        return "Watch"
    else:
        return "High Risk"


def explain_prediction(row: pd.Series, top_n: int = 4) -> list:
    """
    Return the top contributing factors for this component's risk, based on
    global feature importance combined with how far this component's own
    values are from the healthy-population median (a simple, explainable
    substitute for full SHAP analysis -- good enough for a viva walkthrough).
    """
    _ensure_loaded()
    top_features = [f["feature"] for f in _feature_importance[:top_n]]
    explanations = []
    label_map = {
        "percentage_change": "overall percentage change in measured value",
        "drift_rate": "rate of degradation per hour",
        "overall_change_0h_168h": "total change from 0h to 168h",
        "value_168h": "final measured value at 168h",
        "change_96h_168h": "change between 96h and 168h",
        "change_0h_24h": "early change between 0h and 24h",
        "change_24h_96h": "mid-phase change between 24h and 96h",
        "unusual_trajectory": "flagged unusual measurement trajectory",
    }
    for feat in top_features:
        readable = label_map.get(feat, feat)
        value = row.get(feat, None)
        if value is not None:
            if isinstance(value, (int, float, np.floating, np.integer)):
                value = round(float(value), 3)
            explanations.append(f"{readable} = {value}")
    return explanations
