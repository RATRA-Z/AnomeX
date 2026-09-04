"""
scenario_engine.py
-------------------
"Scenario Engine" layer from the project brief. Runs simple, explainable
what-if simulations on top of the trained model and current predictions.

Honesty note (important for the viva): in this dataset, temperature_C and
voltage_V have almost zero correlation with failure_label (checked during
EDA -> ~0.007 and -0.005). So the temperature scenario is still computed
for real using the trained model, but its impact will legitimately be small.
The scenarios that actually move system health are the ones driven by
drift/degradation, which the model is confident about.
"""

import pandas as pd
import numpy as np

from ml.predict import predict_batch, risk_level_from_score


def scenario_current_conditions(raw_df: pd.DataFrame) -> dict:
    preds = predict_batch(raw_df)
    return {"scenario": "Continue Current Conditions",
            "predicted_system_health": round(float(preds["health_score"].mean()), 1)}


def scenario_temperature_change(raw_df: pd.DataFrame, delta_c: float) -> dict:
    df = raw_df.copy()
    df["temperature_C"] = df["temperature_C"] + delta_c
    preds = predict_batch(df)
    label = f"Reduce Temp by {abs(delta_c):.0f}\u00b0C" if delta_c < 0 else f"Increase Temp by {delta_c:.0f}\u00b0C"
    return {"scenario": label,
            "predicted_system_health": round(float(preds["health_score"].mean()), 1)}


def scenario_remove_top_risky(raw_df: pd.DataFrame, top_n: int = 3) -> dict:
    preds = predict_batch(raw_df)
    remaining = preds.sort_values("health_score", ascending=True).iloc[top_n:]
    removed = preds.sort_values("health_score", ascending=True).iloc[:top_n]
    return {
        "scenario": f"Remove Top {top_n} High Risk Components",
        "predicted_system_health": round(float(remaining["health_score"].mean()), 1),
        "removed_component_ids": removed["component_id"].tolist(),
    }


def scenario_more_components_degrade(raw_df: pd.DataFrame, extra_pct: float = 10.0) -> dict:
    """
    Simulate extra_pct% of currently-normal components starting to degrade,
    by pushing their drift-related features toward the average of the
    already-high-risk population, then re-predicting.
    """
    preds = predict_batch(raw_df)
    df = raw_df.copy()
    df["health_score"] = preds["health_score"]
    df["risk_level"] = preds["risk_level"]

    high_risk_avg = df[df["risk_level"] == "High Risk"][
        ["drift_rate", "percentage_change", "overall_change_0h_168h", "change_96h_168h"]
    ].mean()

    normal_idx = df[df["risk_level"] == "Normal"].index
    n_to_flip = int(len(normal_idx) * (extra_pct / 100))
    flip_idx = np.random.RandomState(42).choice(normal_idx, size=min(n_to_flip, len(normal_idx)), replace=False)

    simulated = raw_df.copy()
    for col in ["drift_rate", "percentage_change", "overall_change_0h_168h", "change_96h_168h"]:
        simulated.loc[flip_idx, col] = high_risk_avg[col]

    new_preds = predict_batch(simulated)
    return {
        "scenario": f"If {extra_pct:.0f}% More Components Degrade",
        "predicted_system_health": round(float(new_preds["health_score"].mean()), 1),
        "components_flipped": int(len(flip_idx)),
    }


def run_all_default_scenarios(raw_df: pd.DataFrame) -> list:
    """The 4 default scenarios shown on the dashboard, matching the project brief example."""
    return [
        scenario_current_conditions(raw_df),
        scenario_temperature_change(raw_df, delta_c=-10),
        scenario_remove_top_risky(raw_df, top_n=3),
        scenario_more_components_degrade(raw_df, extra_pct=10),
    ]
