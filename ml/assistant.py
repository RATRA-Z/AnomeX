"""
assistant.py
------------
"AI Reliability Assistant" layer from the project brief.

IMPLEMENTATION NOTE (read this before the viva):
The project brief shows an LLM (OpenAI/Llama) behind the assistant. This
build does not call any external LLM API -- no API key is required to run
this project. Instead, this module uses intent detection (keyword + regex
matching) combined with the REAL trained model and REAL data to generate
genuinely dynamic, data-grounded answers -- the numbers you see are computed
live, not hardcoded. If you want to plug in a real LLM later, the natural
place is `answer_question()`: detect the intent as before, but instead of
formatting the answer with an f-string, pass the retrieved facts to an LLM
prompt for more natural phrasing (a simple RAG pattern).
"""

import re
import pandas as pd

from ml.predict import (
    explain_prediction,
    risk_level_from_score,
    estimate_failure_horizon,
)

from ml import system_analytics
from ml import scenario_engine


def _find_component_id(question: str, valid_ids: set) -> str:
    """Look for something like C00054 or C184 in the question and match it
    against real component_ids (padding-tolerant)."""
    matches = re.findall(r"\bC0*\d{1,6}\b", question.upper())
    for m in matches:
        num = re.sub(r"\D", "", m)
        padded = "C" + num.zfill(5)
        if padded in valid_ids:
            return padded
    return None


def _extract_number(question: str, default=None):
    nums = re.findall(r"\d+(?:\.\d+)?", question)
    return float(nums[0]) if nums else default


def answer_question(question: str, predictions_df: pd.DataFrame, raw_df: pd.DataFrame) -> str:
    q = question.lower().strip()
    valid_ids = set(predictions_df["component_id"].tolist())

    # ---- Intent: explain a specific component ---------------------------
    comp_id = _find_component_id(question, valid_ids)
    if comp_id:
        row = predictions_df[predictions_df["component_id"] == comp_id].iloc[0]
        raw_row = raw_df[raw_df["component_id"] == comp_id].iloc[0]
        reasons = explain_prediction(raw_row)
        horizon = estimate_failure_horizon(raw_row)
        reason_text = "; ".join(reasons) if reasons else "no strong risk signals found"
        if horizon["status"] == "degrading" and horizon["eta_hours"] is not None:
            eta_text = f" Estimated time to reach the safety limit: ~{horizon['eta_hours']}h."
        elif horizon["status"] == "past_safety_limit":
            eta_text = " It has already crossed its safety limit."
        else:
            eta_text = ""
        return (
            f"Component {comp_id} (Lot {row['lot_id']}, {row['component_type']}) has a health score of "
            f"{row['health_score']}/100 and is classified as **{row['risk_level']}** "
            f"(failure probability: {row['failure_probability']*100:.1f}%). "
            f"Key drivers: {reason_text}.{eta_text}"
        )

    # ---- Intent: highest risk components ---------------------------------
    if any(kw in q for kw in ["highest risk", "riskiest", "top risk", "most risky", "worst component"]):
        n = int(_extract_number(question, default=5))
        top = predictions_df.sort_values("health_score").head(n)
        lines = [
            f"{r['component_id']} (Lot {r['lot_id']}, health {r['health_score']}/100, {r['risk_level']})"
            for _, r in top.iterrows()
        ]
        return f"Top {n} highest-risk components:\n- " + "\n- ".join(lines)

    # ---- Intent: remove top N risky components (scenario) ----------------
    if "remove" in q and ("risk" in q or "risky" in q):
        n = int(_extract_number(question, default=3))
        result = scenario_engine.scenario_remove_top_risky(raw_df, top_n=n)
        return (
            f"If we remove the top {n} highest-risk components "
            f"({', '.join(result['removed_component_ids'])}), the predicted system health "
            f"improves to {result['predicted_system_health']}/100."
        )

    # ---- Intent: more components degrade (scenario) -----------------------
    if "degrade" in q or ("more" in q and "component" in q):
        pct = _extract_number(question, default=10)
        result = scenario_engine.scenario_more_components_degrade(raw_df, extra_pct=pct)
        return (
            f"If {pct:.0f}% more components start degrading, predicted system health drops to "
            f"{result['predicted_system_health']}/100 ({result['components_flipped']} components affected)."
        )

    # ---- Intent: temperature scenario --------------------------------------
    if "temperature" in q or "temp" in q:
        delta = -10 if ("reduce" in q or "lower" in q or "decrease" in q) else 10
        result = scenario_engine.scenario_temperature_change(raw_df, delta_c=delta)
        note = (
            " Note: temperature has very little statistical correlation with failure in this "
            "dataset, so the change is small -- degradation rate is the dominant risk factor here."
        )
        return (
            f"Simulating a {delta:+.0f}\u00b0C temperature change: predicted system health "
            f"becomes {result['predicted_system_health']}/100.{note}"
        )

    # ---- Intent: lot-level issues -------------------------------------------
    if "lot" in q:
        summary = system_analytics.lot_risk_summary(predictions_df)
        worst = summary[:3]
        lines = [
            f"{s['lot_id']}: avg health {s['avg_health_score']}/100, "
            f"{s['high_risk_count']} high-risk of {s['component_count']} components ({s['risk_pct']}% risk)"
            for s in worst
        ]
        return "Lots with the most systemic issues:\n- " + "\n- ".join(lines)

    # ---- Intent: overall system health -----------------------------------
    if "system health" in q or "overall health" in q or "how is the system" in q:
        overall = system_analytics.overall_system_health(predictions_df)
        return (
            f"Overall system health is {overall['system_health_score']}/100 across "
            f"{overall['total_components']} components. {overall['high_risk_components']} are High Risk, "
            f"{overall['watch_components']} are on Watch. Riskiest lot: {overall['riskiest_lot']} "
            f"(avg health {overall['riskiest_lot_score']}/100)."
        )

    # ---- Intent: cluster / pattern questions -------------------------------
    if "cluster" in q or "pattern" in q or "similar" in q:
        clusters = system_analytics.degradation_clusters(predictions_df)
        worst = clusters[0]
        return (
            f"The riskiest degradation cluster has {worst['size']} components (mostly from lot "
            f"{worst['dominant_lot']}), with an average health score of {worst['avg_health_score']}/100 "
            f"and {worst['high_risk_count']} components already High Risk. Sample components: "
            f"{', '.join(worst['sample_component_ids'])}."
        )

    # ---- Fallback ----------------------------------------------------------
    return (
        "I can help with: component lookups (e.g. \"Why was C00054 flagged?\"), "
        "\"Which components are highest risk?\", \"What happens if we remove top 3 risky components?\", "
        "\"Are there any lot-level issues?\", \"What if 10% more components degrade?\", "
        "or \"What's the overall system health?\". Try rephrasing your question using one of these."
    )
