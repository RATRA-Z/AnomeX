"""
ml/drift_prediction.py
========================================================================
AnomeX — Module B: Time-Series Drift Predictor
========================================================================

WHAT THE PROBLEM STATEMENT ASKS FOR
-------------------------------------
"Build a predictive regression model that takes Value_0h and Value_24h
as inputs and forecasts Value_168h. If the predicted 168h drift rate
exceeds a calculated safety slope, the system flags the component for
early rejection."

This is DELIBERATELY a different system from:
  - ml/predict.py            (Sukesh's supervised failure model —
                               untouched, not used here)
  - ml/anomaly_detection.py  (Isolation Forest peer-anomaly detector —
                               untouched, not used here, not merged in)

Module B is a REGRESSION problem: predict a number (value_168h) from
two early readings, then turn that prediction into an early-rejection
decision using a threshold ("safety slope") learned from historical
data — NOT a hand-written formula, and NOT the unrelated 26.x
`safety_thresholds.json` values (see the "SAFETY SLOPE METHODOLOGY"
section below for why those are not reused here).

WHY value_0h / value_24h ONLY (+ two derived features)
---------------------------------------------------------
The PS is explicit: inputs are Value_0h and Value_24h. We stick to
that instead of adding value_96h just because it happens to exist in
the CSV — using it would defeat the purpose of an *early* rejection
system (by 96h you've already burned in the component for 4 days;
the whole point of Module B is to flag risk from the first 24h).

We add two features that are mathematically derived ONLY from
value_0h/value_24h (so they introduce no new information/leakage,
just make the early trend explicit for the model):

    early_change = value_24h - value_0h
    early_slope  = early_change / 24        (units: value per hour)

FEATURES USED (final):
    value_0h, value_24h, early_change, early_slope

TARGET:
    value_168h

DATA LEAKAGE — EXPLICITLY EXCLUDED, EVER
-------------------------------------------
None of the following are ever used as model inputs, because at
prediction time (early in the burn-in process) they are either
unknown or are mathematically derived using the very value we are
trying to predict:

    value_168h, change_96h_168h, overall_change_0h_168h,
    percentage_change, drift_rate,
    failure_label, predicted_label, health_score, risk_level,
    failure_probability

`_validate_no_leakage()` below asserts this at import time as a
guard rail, and `prepare_drift_data()` only ever selects from the
explicit FEATURES list.

MODEL CHOICE (and how it's selected — updated)
-------------------------------------------------
Two candidate models are compared, both tree-based ensembles that
need no feature scaling and stay easy to explain in a viva ("many
small trees voting on a number", vs. "trees built one after another,
each correcting the previous one's mistakes"):

    - RandomForestRegressor(n_estimators=300, random_state=42)
    - GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
      max_depth=3, loss="huber", random_state=42)
      — a conservative, shallow-tree, low-learning-rate configuration
      chosen specifically to avoid overfitting a ~10k-row tabular
      dataset; `loss="huber"` makes it less sensitive to outlier
      components than a plain squared-error loss.

SELECTION METHODOLOGY — never against the held-out test set:
    1. The dataset is split once into train (80%) / test (20%),
       `random_state=42` — this test set is set aside and touched
       exactly ONCE, at the very end, for final reporting.
    2. Both candidates are compared using 5-fold cross-validation
       computed ENTIRELY WITHIN the training split (`KFold(n_splits=5,
       shuffle=True, random_state=42)`). For each fold, a fresh model
       is fit on 4/5 of the training data and scored (MAE, RMSE, R²)
       on the held-back 1/5 — averaged across folds per candidate.
    3. MAE is the PRIMARY selection criterion (this mirrors the
       hackathon PS's own evaluation metric: "mean absolute error
       between predicted and actual Value_168h"). The candidate with
       the lower mean cross-validated MAE is selected.
    4. The selected model type is then re-fit ONCE on the FULL
       training split (all 80%), and evaluated ONCE on the held-out
       test split — those are the official, reported MAE/RMSE/R².
    5. For transparency (and to mirror the manual comparison this
       module's design was validated against), the NON-selected
       candidate is also fit on the full training split and scored
       on the test split, purely for side-by-side reporting. This is
       a single, one-shot evaluation done for visibility — it is
       NEVER fed back into hyperparameters or the selection decision,
       so it does not constitute "tuning against the test set".

SAFETY SLOPE METHODOLOGY (read this before the viva)
--------------------------------------------------------
The PS does not hand us a numeric safety slope, and the existing
`ml/models/safety_thresholds.json` file (TYPE_A: 26.179, TYPE_B:
26.582, TYPE_C: 26.486, _default: 26.326) is NOT reused here as the
drift safety slope. Those numbers came from a different, already
existing part of the system (a component-type-keyed threshold used
by the supervised failure model) and describe a MEASURED VALUE
boundary, not a DRIFT RATE (change-per-hour) boundary — the units and
the concept are different, so silently repurposing them would be a
methodology error, not a shortcut.

Instead, `calculate_safety_slope()` derives the slope directly from
historical data, using ONLY the training split (never the held-out
test set, so no future information leaks into the threshold):

    1. For every TRAINING row, compute the historical/observed drift
       rate using the dataset's own actual 0h→168h values:

           observed_drift_rate = (value_168h - value_0h) / 168

       (168 hours is the full burn-in window described by the PS.)

    2. Take the Nth percentile (default: 95th) of that distribution
       as the safety slope.

       Why the 95th percentile specifically? It gives a boundary that
       is: (a) DATA-DERIVED, not guessed; (b) CONSERVATIVE — only the
       fastest-drifting ~5% of historically observed components sit
       above it, matching the PS's framing of "early rejection" as a
       rare, decisive action rather than a routine one; (c) easy to
       explain to a QA inspector: "this is the drift rate that only
       the most extreme 5% of historical units exceeded."

    3. We deliberately do NOT filter by `failure_label` before taking
       the percentile. If we filtered to only "known good" rows (or
       only "known failed" rows) first, the boundary would really be
       a smuggled-in SUPERVISED failure threshold, not the PS's
       "dynamic drift criterion." Using the full training population
       keeps the slope a genuine statistical property of observed
       component behavior, independent of any failure labelling.

    4. This 168h-based percentile is a stand-in for "what a normal
       historical drift rate looks like" — it is then compared
       against the MODEL'S predicted drift rate (computed the same
       way, from `predicted_value_168h`) for new/unseen components,
       which is exactly the PS's rule:

           predicted_drift_rate > safety_slope  →  EARLY REJECT

`calculate_safety_slope()` also cross-checks its own
`observed_drift_rate` formula against the dataset's existing
`drift_rate` column (on the same training rows) and prints a warning
if they disagree beyond a small numerical tolerance — so if this
dataset's `drift_rate` turns out to use a different convention (e.g.
percentage-based, or a different time base), that will be visible
immediately instead of silently baking in a wrong assumption.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
---------------------------------------------
- Does not touch `failure_model.pkl`, `encoders.pkl`,
  `feature_importance.json`, `metrics.json`, or
  `safety_thresholds.json`.
- Does not modify `ml/predict.py`, `ml/preprocessing.py`,
  `ml/config.py`, `database/database.py`, or
  `ml/anomaly_detection.py`.
- Does not merge anomaly detection or failure prediction into this
  regression model — the three stay architecturally separate, and
  are only combined downstream (e.g. in the Streamlit UI or the
  assistant layer) by calling each module's own functions.
- Does not claim physical/causal explanations — every explanation
  string below is phrased as "the model predicts/observes a pattern",
  never "X causes Y".
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# PATHS (reuse ml/config.py's paths if available, else sane defaults)
# ------------------------------------------------------------------
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

DRIFT_MODEL_PATH = os.path.join(MODELS_DIR, "drift_model.pkl")
DRIFT_METRICS_PATH = os.path.join(MODELS_DIR, "drift_metrics.json")

# ------------------------------------------------------------------
# FEATURE / TARGET / LEAKAGE DEFINITIONS
# ------------------------------------------------------------------
BASE_INPUT_COLUMNS = ["value_0h", "value_24h"]  # required by the PS, verbatim
DERIVED_FEATURES = ["early_change", "early_slope"]
FEATURES = BASE_INPUT_COLUMNS + DERIVED_FEATURES
TARGET = "value_168h"
BURN_IN_HOURS = 168  # full 0h -> 168h window, per the PS

LEAKAGE_COLUMNS = {
    "value_168h",
    "change_96h_168h",
    "overall_change_0h_168h",
    "percentage_change",
    "drift_rate",
    "failure_label",
    "predicted_label",
    "health_score",
    "risk_level",
    "failure_probability",
}

_SAFETY_SLOPE_PERCENTILE = 95  # see docstring above for rationale
_DRIFT_RATE_CONVENTION_TOLERANCE = 1e-3  # for the sanity cross-check

_CV_N_SPLITS = 5  # folds used for training-only model selection
_CV_RANDOM_STATE = 42


def _build_candidate_models(random_state: int = 42) -> dict:
    """
    Fresh, unfitted instances of every candidate regressor considered
    for Module B. Kept in one place so model selection always compares
    the exact same hyperparameters described in the module docstring.
    """
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            loss="huber",
            random_state=random_state,
        ),
    }


def _validate_no_leakage() -> None:
    """Guard rail: fail loudly if a future edit accidentally adds a
    forbidden column to FEATURES."""
    leaked = LEAKAGE_COLUMNS.intersection(FEATURES)
    assert not leaked, f"Data leakage rule violated — leaked features: {leaked}"


_validate_no_leakage()


# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
def load_drift_dataset(path: str = DATASET_CSV) -> pd.DataFrame:
    """
    Load the AnomeX dataset and do basic validation that the columns
    Module B depends on are present and usable.
    """
    df = pd.read_csv(path)

    required = set(BASE_INPUT_COLUMNS + [TARGET, "component_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in BASE_INPUT_COLUMNS + [TARGET]:
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains NaNs — Module B expects a clean dataset "
                "(the brief states the dataset has no nulls; re-check the CSV)."
            )

    return df


# ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
def prepare_drift_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build the legal (leakage-free) feature matrix and target.

    Returns:
        X        : DataFrame with columns FEATURES
        y         : Series, the value_168h target
        df_with_features : the input df with early_change/early_slope
                            columns added (handy for downstream safety
                            slope / evaluation calculations that also
                            need value_168h alongside the features)
    """
    out = df.copy()
    out["early_change"] = out["value_24h"] - out["value_0h"]
    out["early_slope"] = out["early_change"] / 24.0

    _validate_no_leakage()
    X = out[FEATURES].copy()
    y = out[TARGET].copy()
    return X, y, out


# ------------------------------------------------------------------
# Bundle: everything needed to predict + explain, saved together
# ------------------------------------------------------------------
@dataclass
class DriftModelBundle:
    model: object
    feature_names: list = field(default_factory=lambda: list(FEATURES))
    safety_slope: float = 0.0
    safety_slope_meta: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    random_state: int = 42
    test_size: float = 0.2


# ------------------------------------------------------------------
# 8. SAFETY SLOPE (training data only — see module docstring)
# ------------------------------------------------------------------
def calculate_safety_slope(
    training_df: pd.DataFrame, percentile: float = _SAFETY_SLOPE_PERCENTILE
) -> tuple[float, dict]:
    """
    Compute the drift safety slope from TRAINING rows only.

    observed_drift_rate = (value_168h - value_0h) / 168   [per training row]
    safety_slope         = the `percentile`-th percentile of that
                            distribution across all training rows

    failure_label is NOT used to filter rows first (see module
    docstring: "SAFETY SLOPE METHODOLOGY" for why). This keeps the
    slope an unsupervised, purely statistical property of historical
    drift behavior.

    Also cross-checks this formula against the dataset's own
    `drift_rate` column (if present) on the same rows, and warns if
    they disagree — protecting against silently assuming the wrong
    convention for this dataset.

    Returns (safety_slope, metadata_dict) where metadata_dict records
    exactly how the number was produced (for QA/judge transparency).
    """
    observed_drift = (training_df["value_168h"] - training_df["value_0h"]) / BURN_IN_HOURS
    safety_slope = float(np.percentile(observed_drift, percentile))

    meta = {
        "method": "percentile_of_historical_observed_drift_rate",
        "formula": "(value_168h - value_0h) / 168",
        "percentile_used": percentile,
        "rows_used": int(len(training_df)),
        "rows_included": "ALL training rows (failure_label NOT used to filter, "
                          "to keep this an unsupervised/data-derived criterion "
                          "rather than a smuggled-in failure threshold)",
        "computed_on": "training split only (test set never touched)",
        "safety_slope_value": safety_slope,
    }

    if "drift_rate" in training_df.columns:
        existing = training_df["drift_rate"].to_numpy()
        diff = np.abs(existing - observed_drift.to_numpy())
        mean_abs_diff = float(np.mean(diff))
        meta["cross_check_vs_existing_drift_rate_column"] = {
            "mean_absolute_difference": mean_abs_diff,
            "matches_expected_convention": bool(
                mean_abs_diff <= _DRIFT_RATE_CONVENTION_TOLERANCE
            ),
        }
        if mean_abs_diff > _DRIFT_RATE_CONVENTION_TOLERANCE:
            warnings.warn(
                "The dataset's existing 'drift_rate' column does not match the "
                "(value_168h - value_0h) / 168 convention used here "
                f"(mean abs difference = {mean_abs_diff:.6f}). Module B's own "
                "observed_drift_rate/safety_slope is still computed consistently "
                "from raw values and remains valid, but double-check the "
                "dataset's drift_rate definition before comparing the two "
                "directly in a report.",
                stacklevel=2,
            )

    return safety_slope, meta


# ------------------------------------------------------------------
# MODEL SELECTION — cross-validated within the TRAINING split only
# ------------------------------------------------------------------
def _cross_validate_candidate(
    model_template, X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """
    5-fold CV for one candidate model, using only the training split
    (the held-out test set is never touched here). A fresh clone of
    `model_template` is fit on each fold so no fitted state leaks
    between folds.
    """
    kf = KFold(n_splits=_CV_N_SPLITS, shuffle=True, random_state=_CV_RANDOM_STATE)
    fold_mae, fold_rmse, fold_r2 = [], [], []

    for fold_train_idx, fold_val_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[fold_train_idx]
        y_fold_train = y_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_val = y_train.iloc[fold_val_idx]

        fold_model = clone(model_template)
        fold_model.fit(X_fold_train, y_fold_train)
        preds = fold_model.predict(X_fold_val)

        fold_mae.append(mean_absolute_error(y_fold_val, preds))
        fold_rmse.append(np.sqrt(mean_squared_error(y_fold_val, preds)))
        fold_r2.append(r2_score(y_fold_val, preds))

    return {
        "mae": float(np.mean(fold_mae)),
        "rmse": float(np.mean(fold_rmse)),
        "r2": float(np.mean(fold_r2)),
    }


def select_drift_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[str, dict]:
    """
    Compare candidate regressors using 5-fold CV computed strictly
    within the training split, and pick the one with the lowest mean
    CV MAE (the PS's own evaluation metric). Never looks at the test
    set — see the module docstring's "MODEL CHOICE" section.

    Returns (selected_model_name, cv_results_by_model).
    """
    candidates = _build_candidate_models(random_state=random_state)
    cv_results = {}
    for name, template in candidates.items():
        cv_results[name] = _cross_validate_candidate(template, X_train, y_train)
        print(
            f"[select_drift_model] {name} 5-fold CV (train-only) | "
            f"MAE={cv_results[name]['mae']:.4f} "
            f"RMSE={cv_results[name]['rmse']:.4f} "
            f"R2={cv_results[name]['r2']:.4f}"
        )

    selected_name = min(cv_results, key=lambda name: cv_results[name]["mae"])
    print(
        f"[select_drift_model] Selected '{selected_name}' "
        f"(lowest cross-validated MAE)."
    )
    return selected_name, cv_results


# ------------------------------------------------------------------
# 3. TRAIN
# ------------------------------------------------------------------
def train_drift_model(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DriftModelBundle, pd.DataFrame, pd.DataFrame]:
    """
    Train Module B's drift predictor:

      1. Split once into train/test (80/20, random_state=42) — the
         test set is set aside and used exactly once, at the end.
      2. Select the best candidate model using 5-fold CV computed
         ONLY within the training split (see select_drift_model()).
      3. Re-fit the selected model type on the FULL training split.
      4. Evaluate ONCE on the held-out test split — these are the
         official, reported metrics.
      5. Also fit + score the non-selected candidate on the same
         split, purely for a one-shot side-by-side comparison report
         (never used to change the selection or hyperparameters).
      6. Compute the safety slope from the training split only.

    Returns (bundle, train_df_with_features, test_df_with_features) —
    the two dataframes are returned so the self-test / evaluation
    functions can reuse the exact same split without recomputing it
    (and accidentally introducing a different random split).
    """
    if df is None:
        df = load_drift_dataset()

    X, y, df_features = prepare_drift_data(df)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_features.index, test_size=test_size, random_state=random_state
    )
    train_df = df_features.loc[train_idx]
    test_df = df_features.loc[test_idx]

    # --- 1. select model via training-only cross-validation ---
    selected_name, cv_results = select_drift_model(
        X_train, y_train, random_state=random_state
    )

    # --- 2. fit every candidate once on the FULL training split, ---
    #        evaluate once on the held-out test split (one-shot,
    #        reporting only — the selection above already happened)
    candidates = _build_candidate_models(random_state=random_state)
    test_results = {}
    fitted_models = {}
    for name, template in candidates.items():
        fitted = clone(template)
        fitted.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        test_results[name] = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
        }
        fitted_models[name] = fitted

    model = fitted_models[selected_name]
    mae = test_results[selected_name]["mae"]
    rmse = test_results[selected_name]["rmse"]
    r2 = test_results[selected_name]["r2"]

    safety_slope, safety_meta = calculate_safety_slope(train_df)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features": FEATURES,
        "target": TARGET,
        "model_type": selected_name,
        "selected_model": selected_name,
        "selection_method": "5-fold CV on training split only, lowest mean MAE",
        "cv_results": cv_results,
        "test_results_all_candidates": test_results,
    }

    bundle = DriftModelBundle(
        model=model,
        feature_names=FEATURES,
        safety_slope=safety_slope,
        safety_slope_meta=safety_meta,
        metrics=metrics,
        random_state=random_state,
        test_size=test_size,
    )

    print(
        f"[train_drift_model] Selected model: {selected_name} | "
        f"trained on {metrics['n_train']} rows, tested on {metrics['n_test']} rows | "
        f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}"
    )
    print(
        f"[train_drift_model] Safety slope (train-only, "
        f"{_SAFETY_SLOPE_PERCENTILE}th percentile of historical drift): "
        f"{safety_slope:.6f}"
    )

    return bundle, train_df, test_df


# ------------------------------------------------------------------
# 4 & 5. SAVE / LOAD
# ------------------------------------------------------------------
def save_drift_model(bundle: DriftModelBundle, metadata: Optional[dict] = None) -> None:
    """
    Save the trained model to ml/models/drift_model.pkl and all
    metadata (feature names, safety slope + its methodology, metrics)
    to ml/models/drift_metrics.json — never touching any other file
    in ml/models/.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(bundle.model, DRIFT_MODEL_PATH)

    payload = {
        "feature_names": bundle.feature_names,
        "target": TARGET,
        "safety_slope": bundle.safety_slope,
        "safety_slope_meta": bundle.safety_slope_meta,
        "metrics": bundle.metrics,
        "random_state": bundle.random_state,
        "test_size": bundle.test_size,
    }
    if metadata:
        payload["extra_metadata"] = metadata

    with open(DRIFT_METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[save_drift_model] Saved model to {DRIFT_MODEL_PATH}")
    print(f"[save_drift_model] Saved metadata to {DRIFT_METRICS_PATH}")


def load_drift_model() -> DriftModelBundle:
    """Reload a previously-saved model + its metadata into a bundle."""
    if not (os.path.exists(DRIFT_MODEL_PATH) and os.path.exists(DRIFT_METRICS_PATH)):
        raise FileNotFoundError(
            "No trained drift model found. Run train_drift_model() + "
            f"save_drift_model() first (expected {DRIFT_MODEL_PATH} and "
            f"{DRIFT_METRICS_PATH})."
        )
    model = joblib.load(DRIFT_MODEL_PATH)
    with open(DRIFT_METRICS_PATH) as f:
        payload = json.load(f)

    return DriftModelBundle(
        model=model,
        feature_names=payload["feature_names"],
        safety_slope=payload["safety_slope"],
        safety_slope_meta=payload["safety_slope_meta"],
        metrics=payload["metrics"],
        random_state=payload.get("random_state", 42),
        test_size=payload.get("test_size", 0.2),
    )


# ------------------------------------------------------------------
# 6. PREDICT value_168h FROM EARLY READINGS ONLY
# ------------------------------------------------------------------
def predict_168h(
    value_0h: float, value_24h: float, bundle: Optional[DriftModelBundle] = None
) -> float:
    """
    Predict value_168h from value_0h and value_24h ONLY — this is the
    literal "future prediction" use case: no value_96h, no value_168h,
    no failure/label information, ever.
    """
    if bundle is None:
        bundle = load_drift_model()

    early_change = value_24h - value_0h
    early_slope = early_change / 24.0
    X = pd.DataFrame(
        [[value_0h, value_24h, early_change, early_slope]], columns=FEATURES
    )
    return float(bundle.model.predict(X)[0])


# ------------------------------------------------------------------
# 7. PREDICTED DRIFT RATE
# ------------------------------------------------------------------
def calculate_predicted_drift(
    value_0h: float, predicted_value_168h: float, hours: int = BURN_IN_HOURS
) -> float:
    """
    predicted_drift_rate = (predicted_value_168h - value_0h) / hours

    Same convention as the training-data-derived observed_drift_rate
    used in calculate_safety_slope(), so the two are directly
    comparable.
    """
    return (predicted_value_168h - value_0h) / hours


def _drift_status(predicted_drift_rate: float, safety_slope: float) -> tuple[bool, str]:
    flag = predicted_drift_rate > safety_slope
    status = "EARLY REJECT" if flag else "SAFE"
    return flag, status


# ------------------------------------------------------------------
# 11. EXPLANATION (interpretation, not causal claim)
# ------------------------------------------------------------------
def explain_drift_prediction(
    value_0h: float,
    value_24h: float,
    predicted_value_168h: float,
    predicted_drift_rate: float,
    safety_slope: float,
    drift_status: str,
) -> str:
    """
    Plain-language explanation suitable for a QA inspector. Describes
    the pattern the model predicted — it does not claim the early
    reading physically *causes* the 168h outcome.
    """
    trend = "increased" if value_24h >= value_0h else "decreased"
    comparison = "above" if drift_status == "EARLY REJECT" else "below"

    return (
        f"Value {trend} from {value_0h:.3f} at 0h to {value_24h:.3f} at 24h. "
        f"Based on this early trajectory, AnomeX predicts a 168h value of "
        f"{predicted_value_168h:.3f}. The predicted drift rate "
        f"({predicted_drift_rate:.6f} per hour) is {comparison} the calculated "
        f"safety slope ({safety_slope:.6f} per hour), so the component is "
        f"classified {drift_status}. This is a statistical pattern learned "
        f"from historical data, not a claim of physical cause."
    )


# ------------------------------------------------------------------
# 9. SINGLE-COMPONENT PREDICTION
# ------------------------------------------------------------------
def predict_component_drift(
    component_id: str,
    df: Optional[pd.DataFrame] = None,
    bundle: Optional[DriftModelBundle] = None,
    include_actual: bool = True,
) -> dict:
    """
    Predict drift outcome for one component.

    The model itself is ALWAYS fed only value_0h/value_24h (+derived
    features) — this mirrors a genuine future prediction, where
    value_168h is not yet known. `include_actual=True` additionally
    attaches the real value_168h (and prediction_error) for
    RETROSPECTIVE evaluation of historical/labelled rows only — set
    it to False to simulate a true "unknown future" scenario.

    Looks the component up via the existing database module if
    available, falling back to the CSV so this stays usable standalone.
    """
    if bundle is None:
        bundle = load_drift_model()

    row = None

    if df is None:
        # Try the existing database module first. get_component()
        # returns a DataFrame (possibly empty, not None, on a miss) —
        # handle that shape explicitly rather than assuming a dict.
        try:
            from database.database import get_component  # existing, not modified

            record = get_component(component_id)
            if record is not None:
                if isinstance(record, pd.DataFrame):
                    if not record.empty:
                        row = record.iloc[0]
                else:
                    # Defensive fallback in case get_component() ever
                    # returns a dict/Series-like object instead.
                    row = pd.Series(record)
        except Exception:
            # Any DB issue (not installed, not configured, lookup
            # error) falls through to the CSV lookup below rather
            # than crashing the prediction.
            row = None

        if row is None:
            df = load_drift_dataset()

    if row is None:
        matches = df[df["component_id"] == component_id]
        if matches.empty:
            raise ValueError(f"component_id '{component_id}' not found in database or dataset.")
        row = matches.iloc[0]

    value_0h = float(row["value_0h"])
    value_24h = float(row["value_24h"])

    predicted_value_168h = predict_168h(value_0h, value_24h, bundle=bundle)
    predicted_drift_rate = calculate_predicted_drift(value_0h, predicted_value_168h)
    flag, status = _drift_status(predicted_drift_rate, bundle.safety_slope)

    result = {
        "component_id": component_id,
        "value_0h": value_0h,
        "value_24h": value_24h,
        "predicted_value_168h": predicted_value_168h,
        "predicted_drift_rate": predicted_drift_rate,
        "safety_slope": bundle.safety_slope,
        "early_rejection_flag": flag,
        "drift_status": status,
    }

    if include_actual and "value_168h" in row.index and pd.notna(row["value_168h"]):
        actual = float(row["value_168h"])
        result["actual_value_168h"] = actual
        result["prediction_error"] = abs(actual - predicted_value_168h)

    return result


# ------------------------------------------------------------------
# 10. EVALUATION
# ------------------------------------------------------------------
def evaluate_drift_model(
    df: Optional[pd.DataFrame] = None, bundle: Optional[DriftModelBundle] = None
) -> dict:
    """
    Re-derive the same train/test split used at training time (same
    random_state/test_size, stored in the bundle) and report MAE,
    RMSE, R², sample counts, and early-rejection classification
    counts on the TEST set — all safety-slope comparisons use the
    slope that was computed from the TRAINING split only, so no test
    information leaks into the threshold.
    """
    if bundle is None:
        bundle = load_drift_model()
    if df is None:
        df = load_drift_dataset()

    X, y, df_features = prepare_drift_data(df)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_features.index,
        test_size=bundle.test_size, random_state=bundle.random_state,
    )
    test_df = df_features.loc[test_idx]

    preds = bundle.model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    predicted_drift = (preds - test_df["value_0h"].to_numpy()) / BURN_IN_HOURS
    early_reject_mask = predicted_drift > bundle.safety_slope

    evaluation = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_test": int(len(test_df)),
        "n_early_reject": int(early_reject_mask.sum()),
        "n_safe": int((~early_reject_mask).sum()),
        "safety_slope_used": bundle.safety_slope,
    }

    # Retrospective-only comparison: how does the predicted status line
    # up against ACTUAL historical drift on the test set? This is
    # purely descriptive/for-the-demo — it is NOT fed back into the
    # model or the threshold.
    actual_drift = (test_df["value_168h"].to_numpy() - test_df["value_0h"].to_numpy()) / BURN_IN_HOURS
    actual_early_reject_mask = actual_drift > bundle.safety_slope
    agreement = float((early_reject_mask == actual_early_reject_mask).mean())
    evaluation["retrospective_status_agreement_with_actual_drift"] = agreement

    return evaluation


# ------------------------------------------------------------------
# SELF TEST
# ------------------------------------------------------------------
def _run_self_test() -> None:
    print("=" * 70)
    print("AnomeX Drift Prediction — self test")
    print("=" * 70)
    print()

    df = load_drift_dataset()
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print()
    print(f"Features:\n{FEATURES}")
    print()
    print(f"Target:\n{TARGET}")
    print()

    bundle, train_df, test_df = train_drift_model(df)
    print(f"Train rows: {bundle.metrics['n_train']}")
    print(f"Test rows: {bundle.metrics['n_test']}")
    print()

    print("Model comparison (5-fold CV within training data — used for selection):")
    for name, cv in bundle.metrics["cv_results"].items():
        print(f"  {name}:")
        print(f"    MAE:  {cv['mae']:.4f}")
        print(f"    RMSE: {cv['rmse']:.4f}")
        print(f"    R2:   {cv['r2']:.4f}")
    print()
    print("Held-out test set (one-shot reporting only, all candidates):")
    for name, tm in bundle.metrics["test_results_all_candidates"].items():
        print(f"  {name}:")
        print(f"    MAE:  {tm['mae']:.4f}")
        print(f"    RMSE: {tm['rmse']:.4f}")
        print(f"    R2:   {tm['r2']:.4f}")
    print()
    print(f"Selected model: {bundle.metrics['selected_model']} "
          f"({bundle.metrics['selection_method']})")
    print()
    print(f"MAE: {bundle.metrics['mae']:.4f}")
    print(f"RMSE: {bundle.metrics['rmse']:.4f}")
    print(f"R2: {bundle.metrics['r2']:.4f}")
    print()
    print(f"Calculated safety slope: {bundle.safety_slope:.6f}")
    print()

    evaluation = evaluate_drift_model(df, bundle=bundle)
    print("Test-set early rejection:")
    print(f"EARLY REJECT: {evaluation['n_early_reject']}")
    print(f"SAFE: {evaluation['n_safe']}")
    print(
        f"(Retrospective agreement with actual historical drift status on "
        f"the test set: {evaluation['retrospective_status_agreement_with_actual_drift']*100:.1f}% "
        "— descriptive only, not used to tune the model or the threshold.)"
    )
    print()
    print("-" * 70)

    # --- C00054 spotlight ---
    target_id = "C00054"
    print(f"\nComponent {target_id}:")
    try:
        result = predict_component_drift(target_id, df=df, bundle=bundle)
        print(f"Value 0h: {result['value_0h']:.3f}")
        print(f"Value 24h: {result['value_24h']:.3f}")
        if "actual_value_168h" in result:
            print(f"Actual Value 168h: {result['actual_value_168h']:.3f}")
        print(f"Predicted Value 168h: {result['predicted_value_168h']:.3f}")
        if "prediction_error" in result:
            print(f"Absolute error: {result['prediction_error']:.3f}")
        print(f"Predicted drift rate: {result['predicted_drift_rate']:.6f}")
        print(f"Safety slope: {result['safety_slope']:.6f}")
        print(f"Status: {result['drift_status']}")
        print()
        print("Explanation:")
        print(
            explain_drift_prediction(
                result["value_0h"],
                result["value_24h"],
                result["predicted_value_168h"],
                result["predicted_drift_rate"],
                result["safety_slope"],
                result["drift_status"],
            )
        )
    except ValueError as exc:
        print(f"[warn] {exc} — skipping spotlight check.")

    print()
    print("-" * 70)

    # --- persistence checks ---
    save_drift_model(bundle)
    print("[OK] No leakage features used "
          f"(FEATURES={FEATURES} excludes all of LEAKAGE_COLUMNS).")
    print("[OK] Model saved.")

    reloaded = load_drift_model()
    print("[OK] Model reload successful.")

    # determinism check: same inputs -> same prediction, before vs after reload
    check_0h, check_24h = float(df.iloc[0]["value_0h"]), float(df.iloc[0]["value_24h"])
    pred_before = predict_168h(check_0h, check_24h, bundle=bundle)
    pred_after = predict_168h(check_0h, check_24h, bundle=reloaded)
    assert np.isclose(pred_before, pred_after), "Prediction changed after reload!"
    print("[OK] Predictions consistent after reload.")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Selected model: {bundle.metrics['selected_model']}")
    print(f"MAE: {bundle.metrics['mae']:.4f}")
    print(f"RMSE: {bundle.metrics['rmse']:.4f}")
    print(f"R2: {bundle.metrics['r2']:.4f}")
    print(f"Safety slope: {bundle.safety_slope:.6f}")
    print(f"EARLY REJECT count (test set): {evaluation['n_early_reject']}")
    print(f"SAFE count (test set): {evaluation['n_safe']}")
    try:
        c00054 = predict_component_drift(target_id, df=df, bundle=bundle)
        print(f"C00054 predicted Value_168h: {c00054['predicted_value_168h']:.3f}")
        if "actual_value_168h" in c00054:
            print(f"C00054 actual Value_168h: {c00054['actual_value_168h']:.3f}")
        print(f"C00054 predicted drift: {c00054['predicted_drift_rate']:.6f}")
        print(f"C00054 status: {c00054['drift_status']}")
    except ValueError as exc:
        print(f"C00054: {exc}")
    print("=" * 70)
    print("Self test complete.")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()
