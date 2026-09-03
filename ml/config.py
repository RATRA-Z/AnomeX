"""
config.py
---------
Central place for all file paths and project-wide constants.
Every other module imports paths from here instead of hardcoding strings,
so the project stays consistent if folders ever move.
"""

import os


# ---- Base directories -------------------------------------------------

ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(ML_DIR, "models")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


# ---- File paths ---------------------------------------------------------

DATASET_CSV = os.path.join(DATA_DIR, "anomex_dataset.csv")

DATABASE_FILE = os.path.join(
    PROJECT_ROOT,
    "database",
    "anomex.db"
)

MODEL_FILE = os.path.join(MODELS_DIR, "failure_model.pkl")
ENCODER_FILE = os.path.join(MODELS_DIR, "encoders.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
THRESHOLDS_FILE = os.path.join(MODELS_DIR, "safety_thresholds.json")
FEATURE_IMPORTANCE_FILE = os.path.join(
    MODELS_DIR,
    "feature_importance.json"
)


# ---- ML constants ------------------------------------------------------

TARGET_COLUMN = "failure_label"


NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "component_type",
    "manufacturer"
]


BOOLEAN_FEATURES = [
    "unusual_trajectory"
]


ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BOOLEAN_FEATURES
)


RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---- Health score ------------------------------------------------------

# 0 = about to fail
# 100 = perfectly healthy

def probability_to_health_score(
    failure_probability: float
) -> float:
    return round((1 - failure_probability) * 100, 1)