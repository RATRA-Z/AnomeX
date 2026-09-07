"""Session-scoped dataset and analysis lifecycle for the AnomeX frontend.

This controller is the single owner of Streamlit session-state entries for
datasets and their ML results.  It does not render UI, write uploaded data to
disk, or modify database state.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, TextIO

import pandas as pd
import streamlit as st

from ml.anomaly_detection import detect_anomalies
from ml.drift_prediction import FEATURES, load_drift_model
from ml.predict import predict_batch
from ml.system_analysis import calculate_component_risk

from .validation import DatasetValidationResult, validate_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "anomex_dataset.csv"
DEFAULT_DATASET_FILENAME = DEFAULT_DATASET_PATH.name

RAW_DATASET_KEY = "raw_dataset"
ANALYZED_DATASET_KEY = "analyzed_dataset"
VALIDATION_RESULT_KEY = "validation_result"
ANALYSIS_METADATA_KEY = "analysis_metadata"
DATASET_SOURCE_KEY = "dataset_source"
DATASET_FILENAME_KEY = "dataset_filename"
ANALYSIS_READY_KEY = "analysis_ready"
ANALYSIS_ERROR_KEY = "analysis_error"

SESSION_DEFAULTS: dict[str, Any] = {
    RAW_DATASET_KEY: None,
    ANALYZED_DATASET_KEY: None,
    VALIDATION_RESULT_KEY: None,
    ANALYSIS_METADATA_KEY: None,
    DATASET_SOURCE_KEY: None,
    DATASET_FILENAME_KEY: None,
    ANALYSIS_READY_KEY: False,
    ANALYSIS_ERROR_KEY: None,
}

DatasetInput = pd.DataFrame | str | Path | PathLike[str] | BinaryIO | TextIO


def initialize_session_state(state: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    """Ensure every controller key exists and return the selected state store.

    Passing a normal mutable mapping makes this module straightforward to test
    outside a running Streamlit application.  Production callers normally omit
    it, which uses ``st.session_state``.
    """
    store = st.session_state if state is None else state
    for key, value in SESSION_DEFAULTS.items():
        store.setdefault(key, value)
    return store


def initialize_default_dataset(
    state: MutableMapping[str, Any] | None = None,
) -> DatasetValidationResult:
    """Load, validate, and analyze the bundled CSV once per session.

    If any current dataset already exists (including a valid uploaded one), it
    remains untouched.  Repeated calls for the default dataset reuse the
    analysis held in session state rather than running ML again.
    """
    store = initialize_session_state(state)

    if store[RAW_DATASET_KEY] is not None:
        result = store[VALIDATION_RESULT_KEY]
        if isinstance(result, DatasetValidationResult):
            return result
        return validate_dataframe(store[RAW_DATASET_KEY])

    try:
        raw_dataset = pd.read_csv(DEFAULT_DATASET_PATH)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        result = DatasetValidationResult(
            is_valid=False,
            errors=[f"Could not load the default AnomeX dataset: {exc}"],
        )
        store[VALIDATION_RESULT_KEY] = result
        store[ANALYSIS_ERROR_KEY] = result.errors[0]
        return result

    result = validate_dataframe(raw_dataset)
    store[VALIDATION_RESULT_KEY] = result
    if not result.is_valid:
        store[ANALYSIS_ERROR_KEY] = "Default dataset validation failed. " + " ".join(result.errors)
        return result

    _set_current_dataset(
        store,
        raw_dataset,
        source="default",
        filename=DEFAULT_DATASET_FILENAME,
        validation_result=result,
    )
    run_analysis(state=store)
    return result


def accept_uploaded_dataset(
    source: DatasetInput,
    filename: str | None = None,
    state: MutableMapping[str, Any] | None = None,
) -> DatasetValidationResult:
    """Validate an upload and, if valid, make it the current unanalyzed dataset.

    Uploaded data is kept only in the supplied session-state mapping.  Invalid
    uploads never replace the current raw or analyzed dataset and never invoke
    ML.  The UI must call :func:`run_analysis` explicitly after a valid upload.
    """
    store = initialize_session_state(state)
    try:
        raw_dataset = source.copy(deep=True) if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        result = DatasetValidationResult(
            is_valid=False,
            errors=[f"Could not read the uploaded CSV: {exc}"],
        )
        store[VALIDATION_RESULT_KEY] = result
        return result

    result = validate_dataframe(raw_dataset)
    store[VALIDATION_RESULT_KEY] = result
    if not result.is_valid:
        return result

    inferred_filename = filename or _source_filename(source) or "uploaded_dataset.csv"
    _set_current_dataset(
        store,
        raw_dataset,
        source="uploaded",
        filename=inferred_filename,
        validation_result=result,
    )
    return result


def run_analysis(
    state: MutableMapping[str, Any] | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame | None:
    """Run the existing component-level ML pipeline once for the current data.

    The result is stored in session state and reused on later calls unless
    ``force=True`` is explicitly supplied.  Exceptions are retained as a
    user-readable state value rather than being swallowed silently.
    """
    store = initialize_session_state(state)
    raw_dataset = store[RAW_DATASET_KEY]
    if not isinstance(raw_dataset, pd.DataFrame):
        store[ANALYSIS_ERROR_KEY] = "No dataset is available to analyze."
        store[ANALYSIS_READY_KEY] = False
        return None

    validation_result = validate_dataframe(raw_dataset)
    store[VALIDATION_RESULT_KEY] = validation_result
    if not validation_result.is_valid:
        store[ANALYSIS_ERROR_KEY] = "Dataset validation failed. " + " ".join(validation_result.errors)
        store[ANALYSIS_READY_KEY] = False
        return None

    if (
        not force
        and store[ANALYSIS_READY_KEY]
        and isinstance(store[ANALYZED_DATASET_KEY], pd.DataFrame)
    ):
        return store[ANALYZED_DATASET_KEY]

    started = perf_counter()
    try:
        analyzed = predict_batch(raw_dataset)
        analyzed = detect_anomalies(analyzed)
        analyzed["predicted_168h"] = _predict_168h_batch(analyzed)
        analyzed = calculate_component_risk(analyzed)
    except Exception as exc:
        error = f"Analysis failed ({type(exc).__name__}): {exc}"
        store[ANALYZED_DATASET_KEY] = None
        store[ANALYSIS_READY_KEY] = False
        store[ANALYSIS_ERROR_KEY] = error
        store[ANALYSIS_METADATA_KEY] = _analysis_metadata(
            store,
            validation_result.row_count,
            succeeded=False,
            elapsed_seconds=perf_counter() - started,
            error=error,
        )
        return None

    elapsed_seconds = perf_counter() - started
    store[ANALYZED_DATASET_KEY] = analyzed
    store[ANALYSIS_READY_KEY] = True
    store[ANALYSIS_ERROR_KEY] = None
    store[ANALYSIS_METADATA_KEY] = _analysis_metadata(
        store,
        len(analyzed),
        succeeded=True,
        elapsed_seconds=elapsed_seconds,
    )
    return analyzed


def get_analyzed_dataset(state: MutableMapping[str, Any] | None = None) -> pd.DataFrame | None:
    """Return the current analyzed dataframe, or ``None`` before success."""
    return initialize_session_state(state)[ANALYZED_DATASET_KEY]


def get_raw_dataset(state: MutableMapping[str, Any] | None = None) -> pd.DataFrame | None:
    """Return the current raw dataframe held in session memory."""
    return initialize_session_state(state)[RAW_DATASET_KEY]


def get_validation_result(
    state: MutableMapping[str, Any] | None = None,
) -> DatasetValidationResult | None:
    """Return the most recent validation result."""
    return initialize_session_state(state)[VALIDATION_RESULT_KEY]


def get_analysis_metadata(state: MutableMapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Return metadata for the latest attempted analysis."""
    return initialize_session_state(state)[ANALYSIS_METADATA_KEY]


def reset_to_default_dataset(state: MutableMapping[str, Any] | None = None) -> DatasetValidationResult:
    """Discard in-memory session data and restore/analyze the bundled CSV."""
    store = initialize_session_state(state)
    for key, value in SESSION_DEFAULTS.items():
        store[key] = value
    return initialize_default_dataset(state=store)


def _set_current_dataset(
    store: MutableMapping[str, Any],
    dataset: pd.DataFrame,
    *,
    source: str,
    filename: str,
    validation_result: DatasetValidationResult,
) -> None:
    """Replace the session's active dataset after successful validation only."""
    store[RAW_DATASET_KEY] = dataset
    store[ANALYZED_DATASET_KEY] = None
    store[VALIDATION_RESULT_KEY] = validation_result
    store[DATASET_SOURCE_KEY] = source
    store[DATASET_FILENAME_KEY] = filename
    store[ANALYSIS_READY_KEY] = False
    store[ANALYSIS_ERROR_KEY] = None
    store[ANALYSIS_METADATA_KEY] = None


def _predict_168h_batch(dataframe: pd.DataFrame) -> pd.Series:
    """Batch the exact feature construction used by ``predict_168h``.

    ``ml.drift_prediction`` exposes a single-value API, but its loaded bundle
    contains the same sklearn estimator used by that API.  Constructing the
    documented four early-reading features once and calling ``model.predict``
    once preserves the model calculation while avoiding 10,000 Python calls.
    """
    bundle = load_drift_model()
    value_0h = dataframe["value_0h"]
    value_24h = dataframe["value_24h"]
    early_change = value_24h - value_0h
    early_slope = early_change / 24.0
    features = pd.DataFrame(
        {
            "value_0h": value_0h,
            "value_24h": value_24h,
            "early_change": early_change,
            "early_slope": early_slope,
        },
        index=dataframe.index,
    )[FEATURES]
    return pd.Series(bundle.model.predict(features), index=dataframe.index, dtype="float64")


def _analysis_metadata(
    store: MutableMapping[str, Any],
    row_count: int,
    *,
    succeeded: bool,
    elapsed_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    previous = store.get(ANALYSIS_METADATA_KEY) or {}
    return {
        "source": store[DATASET_SOURCE_KEY],
        "filename": store[DATASET_FILENAME_KEY],
        "row_count": row_count,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "succeeded": succeeded,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "execution_count": int(previous.get("execution_count", 0)) + 1,
        "error": error,
    }


def _source_filename(source: DatasetInput) -> str | None:
    """Best-effort filename extraction without depending on Streamlit types."""
    name = getattr(source, "name", None)
    if isinstance(name, str) and name.strip():
        return Path(name).name
    if isinstance(source, (str, Path, PathLike)):
        return Path(source).name
    return None
