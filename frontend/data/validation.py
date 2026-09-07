"""Validation for uploaded AnomeX CSV datasets.

This module deliberately validates only; it does not alter a dataframe,
derive missing features, or call any ML code.  Its schema is sourced from
``ml.config`` so uploads are checked against the same inference contracts as
the existing failure-prediction pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import pandas as pd

from ml.config import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


# ``lot_id`` and ``component_id`` are not failure-model features, but they are
# required by the existing dashboard and system-analysis contracts.
IDENTIFIER_COLUMNS = ("component_id", "lot_id")
REQUIRED_NUMERIC_COLUMNS = tuple(NUMERIC_FEATURES)
REQUIRED_CATEGORICAL_COLUMNS = tuple(CATEGORICAL_FEATURES)
REQUIRED_BOOLEAN_COLUMNS = tuple(BOOLEAN_FEATURES)
REQUIRED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *REQUIRED_NUMERIC_COLUMNS,
    *REQUIRED_CATEGORICAL_COLUMNS,
    *REQUIRED_BOOLEAN_COLUMNS,
)

_BOOLEAN_STRINGS = {"true", "false", "1", "0", "yes", "no", "y", "n"}
_BOOLEAN_NUMBERS = {0, 1}


@dataclass(frozen=True)
class DatasetValidationResult:
    """A UI-neutral, serializable summary of an upload validation run."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    missing_columns: list[str] = field(default_factory=list)
    invalid_columns: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return plain Python values suitable for session state or a UI."""
        return asdict(self)


DatasetInput = pd.DataFrame | str | Path | PathLike[str] | BinaryIO | TextIO


def validate_csv_dataset(source: DatasetInput) -> DatasetValidationResult:
    """Read a CSV source, then validate it as an AnomeX inference dataset.

    ``source`` may be a pandas dataframe, a filesystem path, or a file-like
    object such as Streamlit's uploaded-file object.  CSV read failures are
    reported in the result rather than raised to a caller-facing UI.
    """
    if isinstance(source, pd.DataFrame):
        return validate_dataframe(source)

    try:
        dataframe = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
        return DatasetValidationResult(
            is_valid=False,
            errors=[f"Could not read the uploaded CSV: {exc}"],
        )

    return validate_dataframe(dataframe)


def validate_dataframe(dataframe: pd.DataFrame) -> DatasetValidationResult:
    """Validate an in-memory dataframe without changing it.

    The validator requires every inference feature used by ``ml.predict``,
    including all burn-in measurements and precomputed change/drift fields.
    It intentionally does not calculate any omitted feature: preprocessing and
    model behaviour remain solely the responsibility of the existing ML code.
    """
    errors: list[str] = []
    warnings: list[str] = []
    invalid_columns: dict[str, list[str]] = {}

    if not isinstance(dataframe, pd.DataFrame):
        return DatasetValidationResult(
            is_valid=False,
            errors=["Dataset must be a pandas DataFrame or a readable CSV source."],
        )

    row_count = len(dataframe)
    if row_count == 0:
        return DatasetValidationResult(
            is_valid=False,
            errors=["Dataset is empty. Upload a CSV containing at least one component row."],
            row_count=0,
            missing_columns=sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns)),
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        errors.append(
            "Dataset is missing required column(s): " + ", ".join(missing_columns) + "."
        )

    # Stop column-level checks for absent fields, but still report every
    # present-field issue in a single result so users can fix uploads at once.
    for column in REQUIRED_NUMERIC_COLUMNS:
        if column not in dataframe.columns:
            continue

        values = dataframe[column]
        numeric = pd.to_numeric(values, errors="coerce")
        missing_count = int(values.isna().sum())
        non_numeric_count = int((values.notna() & numeric.isna()).sum())
        non_finite_count = int((numeric.notna() & ~np.isfinite(numeric)).sum())

        reasons: list[str] = []
        if missing_count:
            reasons.append(f"{missing_count} missing value(s)")
        if non_numeric_count:
            reasons.append(f"{non_numeric_count} non-numeric value(s)")
        if non_finite_count:
            reasons.append(f"{non_finite_count} non-finite value(s)")
        if reasons:
            invalid_columns[column] = reasons

    for column in (*IDENTIFIER_COLUMNS, *REQUIRED_CATEGORICAL_COLUMNS):
        if column not in dataframe.columns:
            continue

        values = dataframe[column]
        missing_count = int(values.isna().sum())
        blank_count = int(
            values.notna().sum()
            and values.dropna().astype(str).str.strip().eq("").sum()
        )
        reasons = []
        if missing_count:
            reasons.append(f"{missing_count} missing value(s)")
        if blank_count:
            reasons.append(f"{blank_count} blank value(s)")
        if reasons:
            invalid_columns[column] = reasons

    if "component_id" in dataframe.columns:
        component_ids = dataframe["component_id"]
        usable_ids = component_ids.dropna().astype(str).str.strip()
        duplicate_count = int(usable_ids[usable_ids.ne("")].duplicated(keep=False).sum())
        if duplicate_count:
            invalid_columns.setdefault("component_id", []).append(
                f"{duplicate_count} row(s) with duplicate component_id value(s)"
            )

    for column in REQUIRED_BOOLEAN_COLUMNS:
        if column not in dataframe.columns:
            continue

        invalid_boolean_count = int((~dataframe[column].map(_is_boolean_like)).sum())
        if invalid_boolean_count:
            invalid_columns[column] = [
                f"{invalid_boolean_count} value(s) are not usable boolean/binary values "
                "(accepted: true/false, yes/no, y/n, or 1/0)"
            ]

    if invalid_columns:
        errors.append("Dataset contains missing, invalid, or duplicate required values.")

    if "failure_label" not in dataframe.columns:
        warnings.append(
            "failure_label is absent. This is acceptable for inference; historical-label views will be unavailable."
        )

    return DatasetValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        missing_columns=missing_columns,
        invalid_columns=invalid_columns,
    )


def _is_boolean_like(value: Any) -> bool:
    """Return whether a value is compatible with the existing bool feature."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, np.integer, float, np.floating)):
        return value in _BOOLEAN_NUMBERS
    if isinstance(value, str):
        return value.strip().lower() in _BOOLEAN_STRINGS
    return False


if __name__ == "__main__":
    default_dataset = Path(__file__).resolve().parents[2] / "data" / "anomex_dataset.csv"
    print(validate_csv_dataset(default_dataset).to_dict())
