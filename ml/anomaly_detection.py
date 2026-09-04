"""
ml/anomaly_detection.py
========================================================================
AnomeX — Peer-Aware Anomaly Detection Module
========================================================================

WHY THIS MODULE EXISTS (read this first for the viva)
------------------------------------------------------
AnomeX already has a *supervised* failure-prediction model
(ml/models/failure_model.pkl) that answers:

    "Given this component's readings, how likely is it to fail?"

That model was trained on `failure_label`, so it can only recognize
failure patterns that looked like previously-labelled failures.

This module answers a DIFFERENT question:

    "Does this component behave unusually compared with its peers,
     even if nothing in the datasheet limits was technically violated?"

That is UNSUPERVISED anomaly detection. It never sees `failure_label`
(or any other supervised output such as `health_score`,
`risk_level`, `failure_probability`, `predicted_label`). It is only
allowed to look at raw physical/behavioral measurements. This keeps
the two systems conceptually independent, which is the whole point:
- A component can be NORMAL but ANOMALOUS (odd trajectory, but not
  flagged as high failure risk yet — an early-warning signal).
- A component can be HIGH-RISK but NOT statistically unusual (it
  fits a known, common failure pattern the supervised model has
  seen many times before).
- A component can be BOTH or NEITHER.

WHY ISOLATION FOREST?
----------------------
Isolation Forest is a good fit for this hackathon because it:
- Needs no labels (unsupervised) — matches the requirement above.
- Handles multi-dimensional interactions between degradation
  features (e.g. "value dropped AND drift rate spiked") without us
  having to hand-craft every rule.
- Is fast and deterministic (given a fixed `random_state`) on a
  10,000-row tabular dataset — ideal for a live demo.
- Produces a continuous anomaly score, not just a binary flag, so we
  can rank components by "how weird" they are.

WHY THESE FEATURES?
--------------------
We deliberately focus on *behavioral / degradation* features:

    temperature_C, voltage_V,
    value_0h, value_24h, value_96h, value_168h,
    change_0h_24h, change_24h_96h, change_96h_168h,
    overall_change_0h_168h, percentage_change, drift_rate

We do NOT feed `failure_label`, `predicted_label`, `health_score`,
`risk_level`, or `failure_probability` into the model — those are
outputs/labels, not raw behavior, and using them would be data
leakage into an "unsupervised" detector.

`component_type` / `manufacturer` are NOT one-hot encoded straight
into the Isolation Forest. If we did that, the model could start
flagging components as "anomalous" simply for belonging to a rare
category (e.g. a manufacturer with few units), which is an identity
signal, not a *behavioral* signal. The hackathon brief explicitly
warns against letting categorical identity dominate the anomaly
score — so instead we use `component_type` for PEER-RELATIVE
normalization (see below) and keep it out of the raw feature matrix.

PEER-AWARE DESIGN (requirement 6)
-----------------------------------
The simplest technically-defensible design that avoids training
"thousands of tiny Isolation Forest models" is:

    1. Group rows by `component_type`.
    2. Within each group, convert every numeric feature to a
       ROBUST Z-SCORE: (value - group_median) / group_IQR.
       This expresses "how unusual is this value FOR THIS TYPE OF
       COMPONENT", not "how unusual is this value in the whole
       dataset" — i.e. peer-relative behavior baked directly into
       the feature matrix.
    3. Train ONE global Isolation Forest on these peer-normalized
       features.

This gives us peer-aware detection (a component is judged against
components of its own type) while still only training a SINGLE
model — simple, fast, explainable, and easy to defend in a viva.
(If `component_type` has a group with too few members to compute a
stable median/IQR, we fall back to the global median/IQR for that
group so the pipeline never breaks.)

ANOMALY SCORE (0-100)
-----------------------
`sklearn`'s `IsolationForest.score_samples()` returns higher values
for NORMAL points and lower (more negative) values for ANOMALOUS
points. We:

    1. Take the raw score (`raw_isolation_score`).
    2. Flip its sign so bigger = more anomalous.
    3. Min-max scale it into a fixed 0-100 range using the min/max
       observed on the TRAINING set (stored inside the saved model
       bundle), so scoring is deterministic and stable across runs,
       including on brand-new/unseen rows.
    4. Clip to [0, 100] in case a new row is more extreme than
       anything seen during training.

    0   = very normal (fits the trained population well)
    100 = highly anomalous (statistically extreme vs. peers)

ANOMALY FLAG
-------------
`is_anomaly` comes directly from Isolation Forest's own decision
boundary (`predict() == -1`), governed by the `contamination`
parameter. We default to `contamination=0.05` (~5% of rows), which
is a reasonable, commonly-used starting point for hackathon-scale
unsupervised anomaly detection — small enough to stay meaningful
("anomalous" should be rare), large enough to surface a usable list
for the demo. `train_anomaly_model()` prints the resulting count so
you can sanity-check it against the ~337 rows flagged
`unusual_trajectory == 1` in the dataset (they are NOT expected to
match exactly — `unusual_trajectory` is a different, dataset-provided
signal; Isolation Forest is not told about it and is free to
disagree).

WHAT THIS MODULE DOES NOT DO
------------------------------
- It does not retrain or touch `failure_model.pkl`.
- It does not modify `predict.py`, `preprocessing.py`, `service.py`,
  `scenario_engine.py`, `system_analytics.py`, `assistant.py`, or
  `database.py`.
- It does not create a second database.
- It does not claim Isolation Forest gives a *causal* reason for an
  anomaly — `explain_anomaly()` is an interpretation layer built on
  top of the same peer-relative z-scores, clearly documented as such.
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
# We try to reuse ml/config.py's paths if it defines them, so this
# module stays consistent with the rest of the project. If those
# names don't exist there, we fall back to sensible defaults that
# match the project structure described in the brief. Either way,
# nothing in ml/config.py is modified.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODELS_DIR = os.path.join(_THIS_DIR, "models")
_DEFAULT_DATASET_CSV = os.path.join(
    os.path.dirname(_THIS_DIR), "data", "anomex_dataset.csv"
)

try:
    from ml import config as _config  # type: ignore

    MODELS_DIR = getattr(_config, "MODELS_DIR", _DEFAULT_MODELS_DIR)
    DATASET_CSV = getattr(_config, "DATASET_CSV", _DEFAULT_DATASET_CSV)
except Exception:
    MODELS_DIR = _DEFAULT_MODELS_DIR
    DATASET_CSV = _DEFAULT_DATASET_CSV

ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")
ANOMALY_SCALER_PATH = os.path.join(MODELS_DIR, "anomaly_scaler.pkl")

# ------------------------------------------------------------------
# FEATURE DEFINITIONS
# ------------------------------------------------------------------
# Behavioral / degradation features only. Never includes
# failure_label, predicted_label, health_score, risk_level, or
# failure_probability.
ANOMALY_NUMERIC_FEATURES = [
    "temperature_C",
    "voltage_V",
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
    "change_0h_24h",
    "change_24h_96h",
    "change_96h_168h",
    "overall_change_0h_168h",
    "percentage_change",
    "drift_rate",
]

# Used only for peer-grouping (robust z-score), never fed directly
# into the Isolation Forest as a raw category.
PEER_GROUP_COLUMN = "component_type"

# Columns we carry through into the output for readability/joins,
# if present in the input dataframe.
_PASSTHROUGH_COLUMNS = [
    "component_id",
    "lot_id",
    "component_type",
    "manufacturer",
]

# Explicitly-forbidden columns: never used as anomaly-detection
# features, even if present in the input dataframe.
_FORBIDDEN_FEATURES = {
    "failure_label",
    "predicted_label",
    "health_score",
    "risk_level",
    "failure_probability",
}

_MIN_GROUP_SIZE_FOR_STATS = 10  # below this, fall back to global stats


# ------------------------------------------------------------------
# Internal: peer-relative (robust z-score) feature engineering
# ------------------------------------------------------------------
def _compute_group_stats(df: pd.DataFrame) -> dict:
    """
    Compute per-component_type median and IQR (75th - 25th
    percentile) for every numeric anomaly feature, plus global
    fallback stats for any group that's too small to trust.

    Returns a plain dict so it can be pickled/saved alongside the
    model without depending on any exotic object types.
    """
    stats = {"groups": {}, "global": {}}

    for feat in ANOMALY_NUMERIC_FEATURES:
        col = df[feat].astype(float)
        stats["global"][feat] = {
            "median": float(col.median()),
            "iqr": float(_safe_iqr(col)),
        }

    if PEER_GROUP_COLUMN in df.columns:
        for group_name, group_df in df.groupby(PEER_GROUP_COLUMN):
            group_stats = {}
            use_global = len(group_df) < _MIN_GROUP_SIZE_FOR_STATS
            for feat in ANOMALY_NUMERIC_FEATURES:
                if use_global:
                    group_stats[feat] = stats["global"][feat]
                    continue
                col = group_df[feat].astype(float)
                group_stats[feat] = {
                    "median": float(col.median()),
                    "iqr": float(_safe_iqr(col)),
                }
            stats["groups"][str(group_name)] = group_stats

    return stats


def _safe_iqr(series: pd.Series) -> float:
    """IQR with a small floor so we never divide by zero for a
    feature that happens to be constant within a group."""
    q75, q25 = np.nanpercentile(series, [75, 25])
    iqr = q75 - q25
    return iqr if iqr > 1e-6 else 1.0


def _apply_peer_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Convert each numeric anomaly feature into a robust z-score
    relative to its component_type peer group:

        z = (value - group_median) / group_IQR

    Robust statistics (median/IQR) are used instead of mean/std so
    that a handful of extreme components in a group don't distort
    the "normal" baseline for that group.
    """
    out = pd.DataFrame(index=df.index)
    has_group_col = PEER_GROUP_COLUMN in df.columns

    for feat in ANOMALY_NUMERIC_FEATURES:
        values = df[feat].astype(float).to_numpy()
        medians = np.empty(len(df))
        iqrs = np.empty(len(df))

        if has_group_col:
            group_values = df[PEER_GROUP_COLUMN].astype(str).to_numpy()
            for i, g in enumerate(group_values):
                g_stats = stats["groups"].get(g, stats["global"])
                medians[i] = g_stats[feat]["median"]
                iqrs[i] = g_stats[feat]["iqr"]
        else:
            medians[:] = stats["global"][feat]["median"]
            iqrs[:] = stats["global"][feat]["iqr"]

        out[feat] = (values - medians) / iqrs

    return out


# ------------------------------------------------------------------
# Model bundle (kept together so save/load is a single unit)
# ------------------------------------------------------------------
@dataclass
class AnomalyModelBundle:
    model: IsolationForest
    group_stats: dict
    score_min: float
    score_max: float
    contamination: float
    feature_names: list = field(default_factory=lambda: list(ANOMALY_NUMERIC_FEATURES))


def _validate_features(df: pd.DataFrame) -> None:
    missing = [f for f in ANOMALY_NUMERIC_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot run anomaly detection — missing required columns: {missing}"
        )
    leaked = _FORBIDDEN_FEATURES.intersection(df.columns)
    # Not an error — these columns are allowed to exist in the
    # dataframe (e.g. it came from get_all_components()), we just
    # must never select them as model features. This assertion is a
    # guard against a future edit accidentally adding them to
    # ANOMALY_NUMERIC_FEATURES.
    assert not _FORBIDDEN_FEATURES.intersection(ANOMALY_NUMERIC_FEATURES), (
        "Data-science rule violated: a supervised/output column made it "
        "into ANOMALY_NUMERIC_FEATURES."
    )


# ------------------------------------------------------------------
# 13. TRAINING
# ------------------------------------------------------------------
def train_anomaly_model(
    df: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
    save: bool = True,
) -> AnomalyModelBundle:
    """
    Fit a single global Isolation Forest on peer-normalized
    degradation features.

    Steps:
      1. Validate required columns are present.
      2. Compute per-component_type median/IQR ("peer stats").
      3. Convert every numeric feature to a peer-relative robust
         z-score using those stats.
      4. Fit IsolationForest(contamination=..., random_state=42).
      5. Record the min/max raw score on the training set so future
         calls can deterministically rescale to 0-100.
      6. Optionally persist the bundle to ml/models/.

    `contamination` defaults to 0.05 (~5% of rows). This is a
    reasonable hackathon starting point: high enough to produce a
    usable, demo-able list of anomalies, low enough that "anomalous"
    still means something. Print the resulting count after training
    and adjust if it looks too aggressive or too lax for your data.
    """
    _validate_features(df)

    group_stats = _compute_group_stats(df)
    features = _apply_peer_normalization(df, group_stats)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features)

    raw_scores = model.score_samples(features)
    # Flip sign: higher = more anomalous.
    flipped = -raw_scores
    score_min = float(flipped.min())
    score_max = float(flipped.max())

    bundle = AnomalyModelBundle(
        model=model,
        group_stats=group_stats,
        score_min=score_min,
        score_max=score_max,
        contamination=contamination,
    )

    predicted = model.predict(features)  # -1 = anomaly, 1 = normal
    n_anomalies = int((predicted == -1).sum())
    print(
        f"[train_anomaly_model] Trained on {len(df)} rows | "
        f"contamination={contamination} | "
        f"flagged {n_anomalies} rows as anomalous "
        f"({n_anomalies / len(df) * 100:.2f}%)."
    )

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, ANOMALY_MODEL_PATH)
        joblib.dump(
            {
                "group_stats": group_stats,
                "score_min": score_min,
                "score_max": score_max,
                "contamination": contamination,
                "feature_names": ANOMALY_NUMERIC_FEATURES,
            },
            ANOMALY_SCALER_PATH,
        )
        print(
            f"[train_anomaly_model] Saved model to {ANOMALY_MODEL_PATH} "
            f"and scaler/stats to {ANOMALY_SCALER_PATH}."
        )

    return bundle


def _load_bundle() -> AnomalyModelBundle:
    if not (os.path.exists(ANOMALY_MODEL_PATH) and os.path.exists(ANOMALY_SCALER_PATH)):
        raise FileNotFoundError(
            "No trained anomaly model found. Run train_anomaly_model(df) first "
            f"(expected files at {ANOMALY_MODEL_PATH} and {ANOMALY_SCALER_PATH})."
        )
    model = joblib.load(ANOMALY_MODEL_PATH)
    extras = joblib.load(ANOMALY_SCALER_PATH)
    return AnomalyModelBundle(
        model=model,
        group_stats=extras["group_stats"],
        score_min=extras["score_min"],
        score_max=extras["score_max"],
        contamination=extras["contamination"],
        feature_names=extras.get("feature_names", list(ANOMALY_NUMERIC_FEATURES)),
    )


# ------------------------------------------------------------------
# 14. PREDICTION
# ------------------------------------------------------------------
def predict_anomalies(
    df: pd.DataFrame, bundle: Optional[AnomalyModelBundle] = None
) -> pd.DataFrame:
    """
    Score every row in `df` for anomalousness using an already-fit
    bundle (pass one in, or omit to auto-load the saved model from
    ml/models/).

    Returns a NEW dataframe (input df is never mutated) containing:
      - raw_isolation_score : sklearn's native score_samples() output
      - anomaly_score        : 0-100, normalized, deterministic
      - is_anomaly            : bool, from the model's own threshold
      - anomaly_rank          : 1 = most anomalous row in this batch
    plus the original passthrough/id columns and feature columns.
    """
    _validate_features(df)

    if bundle is None:
        bundle = _load_bundle()

    features = _apply_peer_normalization(df, bundle.group_stats)

    with warnings.catch_warnings():
        # Silence sklearn's InconsistentVersionWarning-style noise if
        # the bundle was trained on a slightly different sklearn
        # point release than it's being scored with, mirroring how
        # the existing failure_model.pkl is already handled.
        warnings.simplefilter("ignore")
        raw_scores = bundle.model.score_samples(features)
        predicted = bundle.model.predict(features)  # -1 anomaly, 1 normal

    flipped = -raw_scores
    span = max(bundle.score_max - bundle.score_min, 1e-9)
    normalized = (flipped - bundle.score_min) / span * 100.0
    normalized = np.clip(normalized, 0, 100)

    result = df.copy()
    result["raw_isolation_score"] = raw_scores
    result["anomaly_score"] = np.round(normalized, 2)
    result["is_anomaly"] = predicted == -1
    result["anomaly_rank"] = (
        result["anomaly_score"].rank(method="min", ascending=False).astype(int)
    )

    return result


# ------------------------------------------------------------------
# 7. TOP-LEVEL CONVENIENCE ENTRY POINT
# ------------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-call convenience wrapper for Streamlit / notebooks:

        anomaly_df = detect_anomalies(df)

    Loads the saved model if one exists; if not, trains one on `df`
    first (and saves it), then scores `df`. Does not mutate `df`.
    """
    try:
        bundle = _load_bundle()
    except FileNotFoundError:
        print("[detect_anomalies] No saved model found — training one now.")
        bundle = train_anomaly_model(df, save=True)

    return predict_anomalies(df, bundle=bundle)


# ------------------------------------------------------------------
# 8. SINGLE COMPONENT LOOKUP
# ------------------------------------------------------------------
def get_component_anomaly(component_id: str, anomaly_df: pd.DataFrame) -> dict:
    """
    Return the anomaly-detection result for a single component_id as
    a plain dict (easy to pass to Streamlit / the AI assistant).

    Raises a clear error if the component isn't in anomaly_df rather
    than silently returning nothing.
    """
    if "component_id" not in anomaly_df.columns:
        raise ValueError("anomaly_df has no 'component_id' column.")

    matches = anomaly_df[anomaly_df["component_id"] == component_id]
    if matches.empty:
        raise ValueError(f"component_id '{component_id}' not found in anomaly_df.")

    row = matches.iloc[0]
    return row.to_dict()


# ------------------------------------------------------------------
# 9. EXPLANATION LAYER (interpretation, not causal ground truth)
# ------------------------------------------------------------------
_EXPLANATION_LABELS = {
    "drift_rate": "unusually {direction} drift rate",
    "percentage_change": "unusually {direction} percentage change",
    "change_96h_168h": "unusually {direction} 96h→168h change",
    "overall_change_0h_168h": "unusually {direction} overall degradation",
    "value_168h": "unusual final (168h) value",
    "value_0h": "unusual starting (0h) value",
    "change_0h_24h": "unusually {direction} early (0h→24h) change",
    "change_24h_96h": "unusually {direction} mid (24h→96h) change",
    "temperature_C": "unusual operating temperature",
    "voltage_V": "unusual operating voltage",
    "value_24h": "unusual 24h reading",
    "value_96h": "unusual 96h reading",
}

_EXPLANATION_Z_THRESHOLD = 2.0  # |z| beyond this is called out by name
_MAX_REASONS = 4


def explain_anomaly(row: pd.Series, group_stats: Optional[dict] = None) -> list:
    """
    Produce a short, human-readable, FEATURE-BASED interpretation of
    why a row scored as anomalous — for example:

        ["unusually high drift rate", "unusually large percentage change"]

    IMPORTANT: Isolation Forest itself does not explain *why* a point
    is isolated — it just measures how easy the point was to isolate
    across random splits. This function is a separate, transparent
    interpretation layer: it recomputes the same peer-relative robust
    z-scores used for training/scoring and reports which individual
    features were most extreme for this row relative to its
    component_type peers. Treat these as suggestive interpretation
    aids for a human reviewer, NOT as a causal explanation.

    Pass `group_stats` (e.g. bundle.group_stats) if you have it —
    otherwise this will load the saved bundle's stats automatically.
    """
    if group_stats is None:
        group_stats = _load_bundle().group_stats

    single_row_df = pd.DataFrame([row])
    z_scores = _apply_peer_normalization(single_row_df, group_stats).iloc[0]

    scored_reasons = []
    for feat, z in z_scores.items():
        if abs(z) >= _EXPLANATION_Z_THRESHOLD and feat in _EXPLANATION_LABELS:
            direction = "high" if z > 0 else "low"
            label = _EXPLANATION_LABELS[feat].format(direction=direction)
            scored_reasons.append((abs(z), label))

    scored_reasons.sort(key=lambda pair: pair[0], reverse=True)
    reasons = [label for _, label in scored_reasons[:_MAX_REASONS]]

    if not reasons:
        reasons = ["No single feature stood out strongly — anomaly reflects a "
                    "combination of moderately unusual values across several "
                    "measurements rather than one extreme outlier."]

    return reasons


# ------------------------------------------------------------------
# 10. INTEGRATION WITH EXISTING (SUPERVISED) PREDICTION
# ------------------------------------------------------------------
def analyze_component(component_id: str, anomaly_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Combine the EXISTING supervised failure-prediction pipeline
    (ml.service.predict_component) with this module's unsupervised
    anomaly detection, without modifying either pipeline.

    Returns a dict with:
      failure_probability, predicted_label, health_score, risk_level
        (from the existing failure model, untouched)
      anomaly_score, is_anomaly, anomaly_rank
        (from this module)
      anomaly_explanation
        (from explain_anomaly)

    If `anomaly_df` (the output of detect_anomalies()/predict_anomalies())
    isn't supplied, this will load the DB via database.database and run
    detect_anomalies() on it.
    """
    # --- existing supervised prediction (untouched) ---
    failure_result = {}
    try:
        from ml.service import predict_component  # existing, not modified

        failure_result = predict_component(component_id) or {}
    except Exception as exc:  # pragma: no cover - defensive, keeps this
        # module usable even if ml.service isn't importable in a given
        # context (e.g. being run standalone/tested outside the app).
        failure_result = {"error": f"Could not run existing predict_component: {exc}"}

    # --- this module's anomaly detection ---
    if anomaly_df is None:
        try:
            from database.database import get_all_components  # existing, not modified

            df = get_all_components()
        except Exception:
            df = pd.read_csv(DATASET_CSV)
        anomaly_df = detect_anomalies(df)

    anomaly_result = get_component_anomaly(component_id, anomaly_df)
    reasons = explain_anomaly(anomaly_df[anomaly_df["component_id"] == component_id].iloc[0])

    combined = {
        "component_id": component_id,
        "failure_probability": failure_result.get("failure_probability"),
        "predicted_label": failure_result.get("predicted_label"),
        "health_score": failure_result.get("health_score"),
        "risk_level": failure_result.get("risk_level"),
        "anomaly_score": anomaly_result.get("anomaly_score"),
        "is_anomaly": anomaly_result.get("is_anomaly"),
        "anomaly_rank": anomaly_result.get("anomaly_rank"),
        "anomaly_explanation": reasons,
    }
    return combined


# ------------------------------------------------------------------
# 15. TEST / DEMO ENTRY POINT
# ------------------------------------------------------------------
def _run_self_test() -> None:
    print("=" * 70)
    print("AnomeX anomaly_detection.py — self test")
    print("=" * 70)

    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    bundle = train_anomaly_model(df, save=True)
    result = predict_anomalies(df, bundle=bundle)

    # --- checks ---
    assert len(result) == len(df), "Row count mismatch after scoring."
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert result["anomaly_score"].isna().sum() == 0, "NaN anomaly scores found."

    # determinism check: re-run scoring, scores must match exactly
    result_2 = predict_anomalies(df, bundle=bundle)
    assert np.allclose(result["anomaly_score"], result_2["anomaly_score"]), (
        "Anomaly scores are not deterministic across repeated calls."
    )

    n_anom = int(result["is_anomaly"].sum())
    pct = n_anom / len(result) * 100
    print(f"[OK] anomaly_score present, no NaNs, deterministic across reruns.")
    print(f"[OK] {n_anom} rows flagged anomalous ({pct:.2f}% of dataset).")

    if "unusual_trajectory" in df.columns:
        overlap = int(
            (result["is_anomaly"] & (df["unusual_trajectory"] == 1)).sum()
        )
        total_traj = int((df["unusual_trajectory"] == 1).sum())
        print(
            f"[info] Overlap with dataset's own unusual_trajectory==1 flag: "
            f"{overlap}/{total_traj} rows also caught by Isolation Forest "
            f"(no overlap was forced/expected)."
        )

    # --- C00054 spotlight ---
    target = "C00054"
    if (result["component_id"] == target).any():
        row = get_component_anomaly(target, result)
        reasons = explain_anomaly(result[result["component_id"] == target].iloc[0])
        print("-" * 70)
        print(f"Component {target} (known extremely high failure risk):")
        print(f"  anomaly_score : {row['anomaly_score']}")
        print(f"  is_anomaly    : {row['is_anomaly']}")
        print(f"  anomaly_rank  : {row['anomaly_rank']} of {len(result)}")
        print(f"  interpretation: {reasons}")
        print(
            "  (Note: the anomaly detector was NOT forced to flag this "
            "component — this is whatever Isolation Forest decided from "
            "raw behavior alone.)"
        )
    else:
        print(f"[warn] {target} not found in dataset — skipping spotlight check.")

    # --- mergeability with failure predictions ---
    try:
        from ml.service import predict_component

        merge_check = predict_component(target)
        print("-" * 70)
        print(f"[OK] Existing predict_component('{target}') still callable "
              f"alongside anomaly output: {merge_check}")
    except Exception as exc:
        print(f"[info] Skipped live merge check with ml.service ({exc}). "
              "This is fine when running this file standalone.")

    print("=" * 70)
    print("Self test complete.")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()