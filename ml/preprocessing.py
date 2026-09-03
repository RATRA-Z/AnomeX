"""
preprocessing.py
-----------------
Shared preprocessing logic used by BOTH train_model.py and predict.py.
Keeping this in one place guarantees that training and inference transform
the data in exactly the same way (a very common bug source otherwise).
"""

import os
import pandas as pd
import joblib

from ml.config import (
    DATASET_CSV,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    ENCODER_FILE,
)


def load_raw_dataset() -> pd.DataFrame:
    """Load the validated AnomeX CSV as-is."""
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_CSV}. "
            "Make sure anomex_dataset.csv is inside data."
        )
    df = pd.read_csv(DATASET_CSV)

    # Defensive cleanup: the source file is already marked VALID / is_valid_for_ml,
    # but we still guard against any unexpected nulls or duplicate IDs so the
    # pipeline never silently trains on bad rows.
    df = df.drop_duplicates(subset="component_id")
    df = df.dropna(subset=ALL_FEATURES + [TARGET_COLUMN])
    return df.reset_index(drop=True)


def build_label_encoders(df: pd.DataFrame) -> dict:
    """Create a simple string -> int mapping for each categorical column."""
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        categories = sorted(df[col].dropna().unique().tolist())
        encoders[col] = {cat: idx for idx, cat in enumerate(categories)}
    return encoders


def save_encoders(encoders: dict):
    os.makedirs(os.path.dirname(ENCODER_FILE), exist_ok=True)
    joblib.dump(encoders, ENCODER_FILE)


def load_encoders() -> dict:
    if not os.path.exists(ENCODER_FILE):
        raise FileNotFoundError(
            f"Encoders not found at {ENCODER_FILE}. Run train_model.py first."
        )
    return joblib.load(ENCODER_FILE)


def encode_categorical(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply saved label encoders. Unknown categories fall back to -1."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        mapping = encoders[col]
        df[col + "_enc"] = df[col].map(mapping).fillna(-1).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Turn a raw dataframe (with the original column names from the CSV)
    into the numeric matrix the model expects.
    """
    df = encode_categorical(df, encoders)

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool).astype(int)

    feature_columns = (
        NUMERIC_FEATURES
        + [c + "_enc" for c in CATEGORICAL_FEATURES]
        + BOOLEAN_FEATURES
    )
    X = df[feature_columns].astype(float)
    return X


def get_feature_column_names() -> list:
    return (
        NUMERIC_FEATURES
        + [c + "_enc" for c in CATEGORICAL_FEATURES]
        + BOOLEAN_FEATURES
    )
