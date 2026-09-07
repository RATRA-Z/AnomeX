# ml/system_analysis.py
"""
========================================================================
AnomeX — System-Level Analysis
========================================================================

This module is the SYSTEM-LEVEL layer of AnomeX: it does not run any
of the three component-level ML systems itself, it CONSUMES their
outputs (where available) and looks for patterns ACROSS components —
lots, clusters, behavioral relationships, and "what if" scenarios.

    1. Dynamic Anomaly Detection   (ml/anomaly_detection.py)
       -> anomaly_score, is_anomaly
    2. Time-Series Drift Prediction (ml/drift_prediction.py)
       -> predicted_value_168h, predicted_drift_rate, drift_status
    3. Failure-Risk Prediction      (ml/predict.py / ml/service.py)
       -> failure_probability, predicted_label, health_score, risk_level

None of those three models are touched, retrained, or reimplemented
here. This file only aggregates/synthesizes their outputs when those
columns happen to be present on the incoming DataFrame, and degrades
gracefully (documented fallbacks) when a column is missing.

------------------------------------------------------------------------
THIS FILE IS A MERGE of two previously-separate files:
    - "system_analytics.py"  (lot/health/KMeans clustering — the
       original, already-wired-into-the-backend implementation)
    - "system_analysis.py"   (Mavleen's component-risk scoring,
       relationship graph, What-If simulations, orchestration)

WHY TWO "RISK" AND TWO "SYSTEM HEALTH" NUMBERS BOTH EXIST
------------------------------------------------------------
The merge deliberately keeps TWO clearly-named, clearly-documented
metrics rather than forcing everything into one, because they answer
genuinely different questions:

  (A) FAILURE-MODEL METRICS — `risk_level`, `health_score`
      These are NOT computed in this file. They are outputs already
      attached to each row by the existing failure-prediction system
      (ml/predict.py / ml/service.py) before it reaches here.
      `overall_system_health()` and `lot_risk_summary()` read these
      columns AS-IS. This is the DASHBOARD HEADLINE metric — the
      "System Health" number a user sees on the main screen — and its
      formula/shape is UNCHANGED from the original system_analytics.py
      so nothing that already depends on it breaks.

  (B) COMPOSITE METRICS — `composite_risk_score`, `composite_risk_level`
      Computed in THIS file (`calculate_component_risk()`), by
      blending anomaly_score + failure_probability + drift/percentage
      change + unusual_trajectory into one 0-100 score per component.
      This is intentionally a DIFFERENT column name from `risk_level`
      so it can never silently overwrite the failure model's own
      `risk_level` column. It exists because the What-If simulations
      in this file change RAW measurements (e.g. "what if these
      components degrade 10% more?") — the failure model is not
      re-run live, so `health_score`/`risk_level` would not react to
      that change at all. The composite score IS recomputed from raw
      features every call, so it DOES react, which is what makes the
      What-If feature meaningful.

      `calculate_system_health()` (the composite, single-float version)
      is therefore used ONLY by the What-If simulations
      (`simulate_component_removal`, `simulate_degradation`) — never
      as a replacement for the dashboard's `overall_system_health()`.
      Every dict returned by a What-If function is explicitly labelled
      with `"metric_basis"` so nobody mistakes a composite number for
      the dashboard headline number.

  This is the same "keep both, name them differently, document why"
  approach used below for clustering (KMeans vs. graph communities):
  two lenses on the data that serve two different purposes, not two
  competing answers to the same question.

FUNCTIONS KEPT FROM THE ORIGINAL system_analytics.py (unchanged
formulas/return shapes, so nothing already wired to them breaks):
    - lot_risk_summary()
    - overall_system_health()
    - degradation_clusters()          (KMeans)

FUNCTIONS KEPT FROM MAVLEEN'S system_analysis.py (renamed columns
only where needed to avoid clashing with the failure model's own
`risk_level`; logic otherwise preserved):
    - calculate_component_risk()
    - calculate_system_health()
    - analyze_lots()
    - build_relationship_graph()
    - detect_relationship_communities()  (renamed from
      detect_degradation_clusters — see "DEGRADATION CLUSTERING vs.
      COMPONENT RELATIONSHIP GRAPH" below for why)
    - run_system_analysis()
    - simulate_component_removal()
    - simulate_degradation()

DATA LEAKAGE / PRODUCTION-SAFETY RULE
----------------------------------------
`failure_label` (the historical ground-truth label) is NEVER used as
a stand-in for a live prediction by default anywhere in this file.
`calculate_component_risk()` and `calculate_system_health()` accept
an explicit, OFF-BY-DEFAULT flag, `use_failure_label_fallback`, that
must be deliberately turned on to let `failure_label` contribute to
the composite score — and even then, only as a documented,
RETROSPECTIVE/historical fallback for demoing on the labelled dataset,
never as part of a genuine unseen-component prediction path. Separately,
lot-level historical failure counts (`historical_failure_count` in
`analyze_lots()`) are reported as plain descriptive statistics about
the past, not fed into any risk/health score.

DEGRADATION CLUSTERING vs. COMPONENT RELATIONSHIP GRAPH
------------------------------------------------------------
Both exist, on purpose, because they answer different questions:
    - `degradation_clusters()` (KMeans): "group components into K
      buckets by numeric similarity in drift_rate/health_score."
      Simple, fast, good for a quick dashboard breakdown.
    - `build_relationship_graph()` + `detect_relationship_communities()`
      (nearest-neighbour graph + community detection): "which SPECIFIC
      components behave like each other, and do they cluster into
      organically-connected groups (possibly sharing a lot/type/
      manufacturer)?" Richer, more expensive, better for root-cause
      exploration on the highest-risk components.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "calculate_component_risk",
    "lot_risk_summary",
    "analyze_lots",
    "overall_system_health",
    "calculate_system_health",
    "degradation_clusters",
    "build_relationship_graph",
    "detect_relationship_communities",
    "detect_degradation_clusters",  # back-compat alias, see below
    "simulate_component_removal",
    "simulate_degradation",
    "run_system_analysis",
]


# ------------------------------------------------------------------
# Small internal helpers
# ------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, columns: list, context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required column(s) {missing}.")


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return df[column] if present (NaNs filled), else a same-length
    Series of `default` — keeps optional-column logic in one place."""
    if column in df.columns:
        return df[column].fillna(default)
    return pd.Series(default, index=df.index)


# ============================================================
# Component/System Risk
# ============================================================
def calculate_component_risk(
    df: pd.DataFrame,
    use_failure_label_fallback: bool = False,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Composite 0-100 risk score per component, blending outputs from
    all three AnomeX ML systems where available:

        composite_risk_score = 0.35 * anomaly
                              + 0.40 * failure_probability
                              + 0.20 * drift_risk
                              + 0.05 * trajectory_risk

    Adds two NEW columns — `composite_risk_score` (0-100) and
    `composite_risk_level` (LOW/MEDIUM/HIGH) — deliberately NOT named
    `risk_score`/`risk_level`, so this never overwrites the existing
    failure-prediction system's own `risk_level` column if it's
    already present on `df`. See the module docstring for why both
    concepts are kept.

    Column fallbacks (all optional, degrades gracefully):
        anomaly_score        -> from ml/anomaly_detection.py, already
                                 on a fixed 0-100 scale, so it is
                                 divided by 100 directly (NOT by the
                                 batch's own max — that would make the
                                 score's meaning shift depending on
                                 which subset of components happens to
                                 be passed in, e.g. one dashboard page
                                 vs. another). If missing entirely,
                                 falls back to a drift_rate-based proxy.
        failure_probability   -> from ml/predict.py, already a 0-1
                                 probability.
        failure_label         -> ONLY used if `use_failure_label_fallback
                                 =True` AND failure_probability is
                                 missing. Off by default — see module
                                 docstring's "DATA LEAKAGE /
                                 PRODUCTION-SAFETY RULE".
        percentage_change      -> relative-to-batch normalization (99th
                                 percentile), since (unlike
                                 anomaly_score) it has no fixed known
                                 range.
        unusual_trajectory     -> dataset-provided early-warning flag.

    Idempotent: if `composite_risk_score`/`composite_risk_level`
    already exist on `df` (e.g. this DataFrame was already scored by
    an earlier call in the same request), this returns a copy as-is
    instead of recomputing — pass `force_recompute=True` to override.
    """
    data = df.copy()

    if (
        not force_recompute
        and "composite_risk_score" in data.columns
        and "composite_risk_level" in data.columns
    ):
        return data

    # --------------------------------------------------------
    # Anomaly contribution
    # --------------------------------------------------------
    if "anomaly_score" in data.columns:
        # anomaly_score is defined (ml/anomaly_detection.py) on a
        # fixed 0-100 scale, so divide by that fixed scale rather
        # than the batch's own max.
        anomaly = (_safe_series(data, "anomaly_score") / 100.0).clip(0, 1)
    else:
        # Fallback for a DataFrame that hasn't been through the
        # anomaly detector yet: use drift magnitude as a rough proxy.
        drift = _safe_series(data, "drift_rate").abs()
        max_drift = drift.quantile(0.99)
        anomaly = (drift / max_drift).clip(0, 1) if max_drift else pd.Series(0.0, index=data.index)

    # --------------------------------------------------------
    # Failure prediction contribution
    # --------------------------------------------------------
    if "failure_probability" in data.columns:
        failure_prob = _safe_series(data, "failure_probability").clip(0, 1)
    elif use_failure_label_fallback and "failure_label" in data.columns:
        # RETROSPECTIVE-ONLY fallback, explicitly opted into — see
        # module docstring. Never the default behavior.
        failure_prob = _safe_series(data, "failure_label").clip(0, 1)
    else:
        failure_prob = pd.Series(0.0, index=data.index)

    # --------------------------------------------------------
    # Drift contribution (relative to this batch — percentage_change
    # has no fixed absolute scale, unlike anomaly_score)
    # --------------------------------------------------------
    percentage_change = _safe_series(data, "percentage_change").abs()
    max_change = percentage_change.quantile(0.99)
    drift_risk = (percentage_change / max_change).clip(0, 1) if max_change else pd.Series(0.0, index=data.index)

    # --------------------------------------------------------
    # Trajectory contribution
    # --------------------------------------------------------
    if "unusual_trajectory" in data.columns:
        trajectory_risk = (
            data["unusual_trajectory"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
        )
    else:
        trajectory_risk = pd.Series(0.0, index=data.index)

    # --------------------------------------------------------
    # Final composite score
    #
    # Higher weight on anomaly/failure because false negatives
    # matter most in this problem.
    # --------------------------------------------------------
    risk = 0.35 * anomaly + 0.40 * failure_prob + 0.20 * drift_risk + 0.05 * trajectory_risk

    data["composite_risk_score"] = (risk * 100).clip(0, 100)
    data["composite_risk_level"] = pd.cut(
        data["composite_risk_score"],
        bins=[-1, 35, 65, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    return data


# ============================================================
# Lot-Level Analytics
# ============================================================
def lot_risk_summary(predictions_df: pd.DataFrame) -> list:
    """
    LEGACY / DASHBOARD lot table. Unchanged from the original
    system_analytics.py: one row per lot, based directly on the
    existing failure-prediction system's outputs (`health_score`,
    `risk_level`, `predicted_label`) — not the composite score.

    Returns a list[dict] (not a DataFrame) to preserve the original
    function's contract for any existing caller.
    """
    _require_columns(
        predictions_df,
        ["lot_id", "component_id", "health_score", "risk_level", "predicted_label"],
        "lot_risk_summary",
    )

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
    summary["risk_pct"] = ((summary["high_risk_count"] / summary["component_count"]) * 100).round(1)
    summary = summary.sort_values("avg_health_score", ascending=True)
    return summary.to_dict(orient="records")


def analyze_lots(df: pd.DataFrame, use_failure_label_fallback: bool = False) -> pd.DataFrame:
    """
    EXTENDED lot analysis based on the COMPOSITE risk score (not the
    failure model's own risk_level — see module docstring). Returns a
    DataFrame (not a list), used internally by run_system_analysis()
    and available for richer dashboard views than lot_risk_summary().

    Historical `failure_label` counts (if the column exists) are
    reported as a plain descriptive statistic
    (`historical_failure_count`) — retrospective information about
    the dataset, not a risk/health input.
    """
    data = calculate_component_risk(df, use_failure_label_fallback=use_failure_label_fallback)
    _require_columns(data, ["lot_id", "component_id"], "analyze_lots")

    agg_kwargs = dict(
        total_components=("component_id", "count"),
        average_composite_risk=("composite_risk_score", "mean"),
        max_composite_risk=("composite_risk_score", "max"),
    )
    if "drift_rate" in data.columns:
        agg_kwargs["average_drift"] = ("drift_rate", "mean")
    if "percentage_change" in data.columns:
        agg_kwargs["average_percentage_change"] = ("percentage_change", "mean")
    if "failure_label" in data.columns:
        # Descriptive/retrospective only — see docstring.
        agg_kwargs["historical_failure_count"] = ("failure_label", "sum")

    lot_analysis = data.groupby("lot_id", observed=True).agg(**agg_kwargs).reset_index()

    high_counts = (
        data[data["composite_risk_level"] == "HIGH"]
        .groupby("lot_id", observed=True)
        .size()
        .rename("high_composite_risk_components")
    )
    lot_analysis = lot_analysis.merge(high_counts, on="lot_id", how="left")
    lot_analysis["high_composite_risk_components"] = (
        lot_analysis["high_composite_risk_components"].fillna(0).astype(int)
    )
    lot_analysis["high_risk_percentage"] = (
        lot_analysis["high_composite_risk_components"] / lot_analysis["total_components"] * 100
    )

    lot_analysis["lot_health"] = (
        100
        - (0.7 * lot_analysis["average_composite_risk"] + 0.3 * lot_analysis["high_risk_percentage"])
    ).clip(0, 100)

    lot_analysis["lot_status"] = pd.cut(
        lot_analysis["lot_health"],
        bins=[-1, 50, 75, 100],
        labels=["CRITICAL", "WATCH", "HEALTHY"],
    )

    return lot_analysis.sort_values("average_composite_risk", ascending=False)


# ============================================================
# Overall System Health
# ============================================================
def overall_system_health(predictions_df: pd.DataFrame) -> dict:
    """
    THE DASHBOARD HEADLINE "System Health" metric. Unchanged from the
    original system_analytics.py: based directly on the existing
    failure-prediction system's outputs (`health_score`, `risk_level`).

    This is the number that should be shown wherever a single "System
    Health" figure is displayed. See module docstring for why this is
    intentionally different from calculate_system_health() below.
    """
    _require_columns(predictions_df, ["risk_level", "health_score", "lot_id"], "overall_system_health")

    total = len(predictions_df)
    high_risk = int((predictions_df["risk_level"] == "High Risk").sum())
    watch = int((predictions_df["risk_level"] == "Watch").sum())
    normal = total - high_risk - watch
    avg_health = round(float(predictions_df["health_score"].mean()), 1)

    riskiest_lot_row = predictions_df.groupby("lot_id")["health_score"].mean().sort_values().head(1)
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


def calculate_system_health(df: pd.DataFrame, use_failure_label_fallback: bool = False) -> float:
    """
    COMPOSITE system health (0-100), recomputed from raw features via
    calculate_component_risk(). Used ONLY by the What-If simulations
    below, which change raw measurements and need a number that
    reacts to that change — NOT a substitute for the dashboard's
    overall_system_health(). See module docstring for the full
    rationale.

    `failure_label`'s contribution (`failure_ratio`) is retrospective/
    historical by nature (it describes the makeup of a labelled
    dataset) and is included in the formula only when
    `use_failure_label_fallback=True`; otherwise that term is simply
    left out of the weighted sum rather than silently substituted.
    """
    data = calculate_component_risk(df, use_failure_label_fallback=use_failure_label_fallback)

    average_risk = data["composite_risk_score"].mean()
    high_risk_ratio = (data["composite_risk_level"] == "HIGH").mean()

    if use_failure_label_fallback and "failure_label" in data.columns:
        failure_ratio = data["failure_label"].mean()
        system_risk = 0.60 * average_risk + 0.25 * high_risk_ratio * 100 + 0.15 * failure_ratio * 100
    else:
        # Renormalized without the failure_label term.
        system_risk = (0.60 / 0.85) * average_risk + (0.25 / 0.85) * high_risk_ratio * 100

    system_health = 100 - system_risk
    return round(float(np.clip(system_health, 0, 100)), 2)


# ============================================================
# Degradation Clustering  (KMeans — numeric similarity buckets)
# ============================================================
def degradation_clusters(predictions_df: pd.DataFrame, n_clusters: int = 6) -> list:
    """
    Groups components by degradation similarity using KMeans on
    [drift_rate, health_score]. Returns clusters sorted by risk
    (lowest avg health first = highest risk).

    Unchanged from the original system_analytics.py. See the module
    docstring ("DEGRADATION CLUSTERING vs. COMPONENT RELATIONSHIP
    GRAPH") for how this differs from detect_relationship_communities()
    below.
    """
    _require_columns(
        predictions_df,
        ["drift_rate", "health_score", "lot_id", "component_id", "risk_level"],
        "degradation_clusters",
    )

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
        clusters.append(
            {
                "cluster_id": int(cid),
                "size": int(len(group)),
                "avg_health_score": round(float(group["health_score"].mean()), 1),
                "avg_drift_rate": round(float(group["drift_rate"].mean()), 4),
                "dominant_lot": top_lot,
                "high_risk_count": int((group["risk_level"] == "High Risk").sum()),
                "sample_component_ids": group["component_id"].head(5).tolist(),
            }
        )

    return sorted(clusters, key=lambda c: c["avg_health_score"])


# ============================================================
# Component Relationship Graph
# ============================================================
def build_relationship_graph(
    df: pd.DataFrame,
    max_components: int = 500,
    neighbours: int = 5,
    similarity_threshold: float = 0.82,
    use_failure_label_fallback: bool = False,
):
    """
    Builds a graph where:
        Node = Component
        Edge = Similar degradation behaviour

    We don't compare all components against each other (expensive);
    instead we focus on the highest-COMPOSITE-risk components and use
    nearest neighbours on their raw measurement/drift features.

    Node/edge attributes use `composite_risk_score`/`composite_risk_level`
    (not `risk_score`/`risk_level`) — see module docstring.
    """
    data = calculate_component_risk(df, use_failure_label_fallback=use_failure_label_fallback)
    data = data.sort_values("composite_risk_score", ascending=False).head(max_components).copy()

    candidate_features = [
        "value_0h", "value_24h", "value_96h", "value_168h",
        "change_0h_24h", "change_24h_96h", "change_96h_168h",
        "drift_rate", "percentage_change",
    ]
    features = [c for c in candidate_features if c in data.columns]
    if not features:
        raise ValueError("build_relationship_graph: none of the expected measurement/drift columns are present.")

    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    n_neighbors = min(neighbours + 1, len(data))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(X_scaled)
    distances, indices = model.kneighbors(X_scaled)

    graph = nx.Graph()

    for _, row in data.iterrows():
        graph.add_node(
            row["component_id"],
            composite_risk_score=round(float(row["composite_risk_score"]), 2),
            composite_risk_level=str(row["composite_risk_level"]),
            lot_id=row.get("lot_id", "UNKNOWN"),
            component_type=row.get("component_type", "UNKNOWN"),
            manufacturer=row.get("manufacturer", "UNKNOWN"),
        )

    for i in range(len(data)):
        component_a = data.iloc[i]["component_id"]
        for j in range(1, n_neighbors):
            neighbour_index = indices[i][j]
            component_b = data.iloc[neighbour_index]["component_id"]
            distance = distances[i][j]

            similarity = 1 / (1 + distance)

            same_lot = data.iloc[i]["lot_id"] == data.iloc[neighbour_index]["lot_id"]
            same_type = data.iloc[i]["component_type"] == data.iloc[neighbour_index]["component_type"]
            same_manufacturer = (
                data.iloc[i]["manufacturer"] == data.iloc[neighbour_index]["manufacturer"]
            )

            metadata_bonus = 0.06 * same_lot + 0.03 * same_type + 0.02 * same_manufacturer
            similarity = min(1, similarity + metadata_bonus)

            if similarity >= similarity_threshold:
                graph.add_edge(
                    component_a,
                    component_b,
                    similarity=round(float(similarity), 3),
                    same_lot=bool(same_lot),
                    same_type=bool(same_type),
                    same_manufacturer=bool(same_manufacturer),
                )

    return graph, data


def detect_relationship_communities(graph: nx.Graph) -> list:
    """
    Finds organically-connected groups of components in the
    RELATIONSHIP GRAPH (build_relationship_graph's output) using
    label-propagation community detection — different in kind from
    degradation_clusters()'s KMeans buckets; see module docstring.

    (Previously named `detect_degradation_clusters` in Mavleen's
    original file — kept available under that name too, below, for
    backward compatibility.)
    """
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return []

    communities = list(nx.community.asyn_lpa_communities(graph, weight="similarity", seed=42))

    clusters = []
    for cluster_id, community in enumerate(communities, start=1):
        nodes = list(community)
        risks = [graph.nodes[n]["composite_risk_score"] for n in nodes]
        lots = [graph.nodes[n]["lot_id"] for n in nodes]
        manufacturers = [graph.nodes[n]["manufacturer"] for n in nodes]

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(nodes),
                "components": nodes,
                "average_composite_risk": round(float(np.mean(risks)), 2),
                "max_composite_risk": round(float(np.max(risks)), 2),
                "dominant_lot": pd.Series(lots).mode().iloc[0],
                "dominant_manufacturer": pd.Series(manufacturers).mode().iloc[0],
            }
        )

    clusters.sort(key=lambda c: c["average_composite_risk"], reverse=True)
    return clusters


def detect_degradation_clusters(graph: nx.Graph) -> list:
    """Back-compat alias for detect_relationship_communities() — kept
    so any existing import of Mavleen's original name keeps working."""
    return detect_relationship_communities(graph)


# ============================================================
# What-If Scenarios
# ============================================================
# NOTE: both scenarios below use calculate_system_health() (the
# COMPOSITE metric), not overall_system_health() (the dashboard
# headline metric) — see module docstring for why. Every result dict
# carries an explicit "metric_basis" key so this is never ambiguous
# to a caller. These are model-based SIMULATIONS, not physical digital
# twins, and do not re-run the anomaly/failure/drift models.

_WHAT_IF_METRIC_BASIS = (
    "composite_risk_score (recomputed from raw drift features each call); "
    "NOT the same figure as overall_system_health()'s dashboard headline "
    "metric, which depends on the failure model's cached outputs."
)


def simulate_component_removal(df: pd.DataFrame, component_ids: list) -> dict:
    """What happens to (composite) system health if the given
    components are rejected/removed? A simulation, not a digital twin."""
    original_health = calculate_system_health(df)
    simulated_df = df[~df["component_id"].isin(component_ids)].copy()
    new_health = calculate_system_health(simulated_df)

    return {
        "scenario": "Remove selected components",
        "metric_basis": _WHAT_IF_METRIC_BASIS,
        "removed_components": component_ids,
        "original_health": original_health,
        "simulated_health": new_health,
        "health_change": round(new_health - original_health, 2),
    }


def simulate_degradation(df: pd.DataFrame, component_ids: list, degradation_percentage: float = 10) -> dict:
    """
    Simulates selected components continuing to degrade by
    `degradation_percentage`, recomputes drift-derived columns, and
    reports the (composite) system health impact.

    IMPORTANT: this is a model-based scenario, NOT a physical digital
    twin — and because the anomaly/failure models are not re-run live,
    only the drift/percentage-change contribution to composite risk
    reflects the simulated change; any cached anomaly_score/
    failure_probability columns are left as-is.
    """
    original_health = calculate_system_health(df)

    simulated = df.copy()
    mask = simulated["component_id"].isin(component_ids)
    factor = 1 + degradation_percentage / 100

    for column in ["value_24h", "value_96h", "value_168h"]:
        if column in simulated.columns:
            simulated.loc[mask, column] *= factor

    if all(col in simulated.columns for col in ["value_0h", "value_168h"]):
        simulated["overall_change_0h_168h"] = simulated["value_168h"] - simulated["value_0h"]
        simulated["percentage_change"] = (
            simulated["overall_change_0h_168h"] / simulated["value_0h"] * 100
        )
        simulated["drift_rate"] = simulated["overall_change_0h_168h"] / 168

    new_health = calculate_system_health(simulated)

    return {
        "scenario": f"{degradation_percentage}% additional degradation",
        "metric_basis": _WHAT_IF_METRIC_BASIS,
        "affected_components": len(component_ids),
        "original_health": original_health,
        "simulated_health": new_health,
        "health_change": round(new_health - original_health, 2),
    }


# ============================================================
# Complete System Analysis
# ============================================================
def run_system_analysis(df: pd.DataFrame, use_failure_label_fallback: bool = False) -> dict:
    """
    Orchestrates the full system-level analysis. Run this after the
    three component-level ML systems have already populated whichever
    of their output columns are available on `df`.

    Returns both the composite view (used for the graph/clusters/lots)
    AND the dashboard's legacy overall_system_health() figure, clearly
    labelled, so downstream consumers can display the right number in
    the right place.
    """
    data = calculate_component_risk(df, use_failure_label_fallback=use_failure_label_fallback)

    composite_health = calculate_system_health(data, use_failure_label_fallback=use_failure_label_fallback)

    dashboard_health = None
    try:
        dashboard_health = overall_system_health(df)
    except ValueError:
        # df doesn't have the failure-model columns yet — the
        # composite view still works without it.
        warnings.warn(
            "run_system_analysis: overall_system_health() could not be computed "
            "(missing failure-model columns) — dashboard_system_health will be None.",
            stacklevel=2,
        )

    lots = analyze_lots(data, use_failure_label_fallback=use_failure_label_fallback)
    graph, graph_data = build_relationship_graph(data, use_failure_label_fallback=use_failure_label_fallback)
    clusters = detect_relationship_communities(graph)

    high_risk = data[data["composite_risk_level"] == "HIGH"]
    medium_risk = data[data["composite_risk_level"] == "MEDIUM"]

    summary = {
        "composite_system_health": composite_health,
        "dashboard_system_health": dashboard_health["system_health_score"] if dashboard_health else None,
        "total_components": len(data),
        "high_risk_components": len(high_risk),
        "medium_risk_components": len(medium_risk),
        "high_risk_percentage": round(len(high_risk) / len(data) * 100, 2) if len(data) else 0.0,
        "number_of_relationships": graph.number_of_edges(),
        "number_of_clusters": len(clusters),
        "highest_risk_lot": lots.iloc[0]["lot_id"] if len(lots) > 0 else None,
    }

    return {
        "components": data,
        "summary": summary,
        "lots": lots,
        "graph": graph,
        "graph_components": graph_data,
        "clusters": clusters,
    }
