"""
system_analytics.py
--------------------
"System Level" layer from the project brief: instead of looking at one
component at a time, this finds patterns ACROSS components.

Two things are computed:
1. Lot-level risk summary (avg health, high-risk count, overall system health)
2. Degradation clusters: components grouped by how similarly they are
   degrading (using KMeans on drift-related features), which mirrors the
   "Degradation Cluster" graph in the project brief.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def lot_risk_summary(predictions_df: pd.DataFrame) -> list:
    """One row per lot: avg health score, high-risk count, failure rate."""
    summary = (
        predictions_df.groupby("lot_id")
        .agg(
            component_count=("component_id", "count"),
            avg_health_score=("health_score", "mean"),
            high_risk_count=("risk_level", lambda s: (s == "High Risk").sum()),
            watch_count=("risk_level", lambda s: (s == "Watch").sum()),
            predicted_failure_count=("predicted_label", "sum"),
        )
        .reset_index()
    )
    summary["avg_health_score"] = summary["avg_health_score"].round(1)
    summary["risk_pct"] = (
        (summary["high_risk_count"] / summary["component_count"]) * 100
    ).round(1)
    summary = summary.sort_values("avg_health_score", ascending=True)
    return summary.to_dict(orient="records")


def overall_system_health(predictions_df: pd.DataFrame) -> dict:
    total = len(predictions_df)
    high_risk = int((predictions_df["risk_level"] == "High Risk").sum())
    watch = int((predictions_df["risk_level"] == "Watch").sum())
    normal = total - high_risk - watch
    avg_health = round(float(predictions_df["health_score"].mean()), 1)

    riskiest_lot_row = (
        predictions_df.groupby("lot_id")["health_score"].mean().sort_values().head(1)
    )
    riskiest_lot = riskiest_lot_row.index[0] if len(riskiest_lot_row) else None
    riskiest_lot_score = round(float(riskiest_lot_row.iloc[0]), 1) if len(riskiest_lot_row) else None

    return {
        "total_components": total,
        "system_health_score": avg_health,
        "high_risk_components": high_risk,
        "watch_components": watch,
        "normal_components": normal,
        "riskiest_lot": riskiest_lot,
        "riskiest_lot_score": riskiest_lot_score,
    }


def degradation_clusters(predictions_df: pd.DataFrame, n_clusters: int = 6) -> list:
    """
    Group components by degradation similarity using KMeans on
    [drift_rate, percentage_change_equivalent(health drop), overall trend].
    Returns clusters sorted by risk (lowest avg health first = highest risk).
    """
    df = predictions_df.copy()
    features = df[["drift_rate", "health_score"]].fillna(0)

    k = min(n_clusters, max(2, df["lot_id"].nunique() // 3))
    if len(df) < k:
        k = max(2, len(df) // 2)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(features)

    clusters = []
    for cid, group in df.groupby("cluster_id"):
        top_lot = group["lot_id"].value_counts().idxmax()
        clusters.append({
            "cluster_id": int(cid),
            "size": int(len(group)),
            "avg_health_score": round(float(group["health_score"].mean()), 1),
            "avg_drift_rate": round(float(group["drift_rate"].mean()), 4),
            "dominant_lot": top_lot,
            "high_risk_count": int((group["risk_level"] == "High Risk").sum()),
            "sample_component_ids": group["component_id"].head(5).tolist(),
        })

    clusters = sorted(clusters, key=lambda c: c["avg_health_score"])
    return clusters
