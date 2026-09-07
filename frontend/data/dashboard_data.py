"""
dashboard_data.py
-----------------
Presentation adapters for an already-analyzed AnomeX dataframe.

ML orchestration belongs to ``data.session``. This module only reads the
unprocessed bundled CSV when explicitly requested; it never calls a model.
"""

from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "anomex_dataset.csv"


def load_dashboard_data(analyzed_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return supplied analyzed data, or load the raw default CSV.

    New callers should pass the dataframe from ``data.session``. The
    no-argument form is retained only for compatibility and performs no ML
    analysis of any kind.
    """
    if analyzed_df is not None:
        _require_dataframe(analyzed_df, "load_dashboard_data")
        return analyzed_df
    return pd.read_csv(DATASET_PATH)


def get_kpi_metrics(df: pd.DataFrame) -> dict:
    """Return headline metrics from analyzed ML outputs only.

    The composite model has LOW/MEDIUM/HIGH levels. It has no separate
    Critical class, so this adapter returns it as unavailable rather than
    silently treating HIGH as Critical.
    """
    if _is_empty(df):
        total = high_risk = medium_risk = 0
        system_health = average_composite_risk = 0.0
    else:
        _require_columns(
            df,
            ["composite_risk_score", "composite_risk_level", "health_score"],
            "get_kpi_metrics",
        )
        total = len(df)
        high_risk = int((df["composite_risk_level"].astype(str) == "HIGH").sum())
        medium_risk = int((df["composite_risk_level"].astype(str) == "MEDIUM").sum())
        system_health = round(float(df["health_score"].mean()), 1)
        average_composite_risk = round(float(df["composite_risk_score"].mean()), 1)

    return {
        "total_components": total,
        "total_components_pct": 100.0 if total else 0.0,
        "high_risk": high_risk,
        "high_risk_pct": round((high_risk / total) * 100, 2)
        if total
        else 0.0,
        "average_composite_risk": average_composite_risk,
        "critical": None,
        "critical_pct": None,
        "critical_available": False,
        "critical_semantics": "No separate Critical ML risk class is defined by the current pipeline.",
        "system_health": system_health,
        "system_health_max": 100,
        "medium_risk": medium_risk,
    }


def get_high_risk_components(
    df: pd.DataFrame,
    limit: int = 5,
) -> pd.DataFrame:
    """Return the highest composite-risk components for the dashboard table.

    ``Anomaly Score`` is a normalized 0-100 detector score, not a probability.
    """
    columns = ["Component ID", "Lot ID", "Health Score", "Anomaly Score", "Predicted 168h", "Risk Level"]
    if _is_empty(df):
        return pd.DataFrame(columns=columns)
    _require_columns(
        df,
        ["component_id", "lot_id", "health_score", "composite_risk_score", "composite_risk_level"],
        "get_high_risk_components",
    )
    if limit < 0:
        raise ValueError("get_high_risk_components: limit must be zero or greater.")

    result = df[df["composite_risk_level"].astype(str) == "HIGH"]
    result = result.sort_values("composite_risk_score", ascending=False).head(limit).copy()

    table = pd.DataFrame(
        {
            "Component ID": result["component_id"],
            "Lot ID": result["lot_id"],
            "Health Score": result["health_score"].round(1),
            "Anomaly Score": _rounded_optional_column(result, "anomaly_score", 2),
            "Predicted 168h": _optional_column(result, "predicted_168h"),
            "Risk Level": result["composite_risk_level"],
        }
    )
    table.attrs["anomaly_score_semantics"] = "Normalized anomaly score on a 0-100 scale; not a probability."
    table.attrs["risk_level_semantics"] = "Composite ML risk level: LOW, MEDIUM, or HIGH."
    return table.reset_index(drop=True)


def get_lot_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a lot summary without inventing a Critical ML risk class.

    ``Critical`` is retained as an all-missing compatibility column. ``Status``
    is derived solely from average health and is explicitly separate from the
    composite risk level/count reported by ``High_Risk``.
    """
    columns = ["Lot ID", "Components", "High_Risk", "Critical", "Avg_Health", "Status"]
    if _is_empty(df):
        return pd.DataFrame(columns=columns)
    _require_columns(
        df,
        ["lot_id", "component_id", "health_score", "composite_risk_level"],
        "get_lot_health_summary",
    )

    result = (
        df.groupby("lot_id")
        .agg(
            Components=("component_id", "count"),
            High_Risk=(
                "composite_risk_level",
                lambda s: (s.astype(str) == "HIGH").sum(),
            ),
            Avg_Health=("health_score", "mean"),
        )
        .reset_index()
    )

    result["Critical"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["Status"] = pd.cut(
        result["Avg_Health"],
        bins=[-np.inf, 50, 75, np.inf],
        labels=["Critical health", "Watch health", "Good health"],
    ).astype(str)

    result = result.rename(columns={"lot_id": "Lot ID"})

    result["Avg_Health"] = result["Avg_Health"].round(1)

    result = result[columns].sort_values("Avg_Health", ascending=True).reset_index(drop=True)
    result.attrs["critical_semantics"] = "Unavailable: the current ML pipeline has no separate Critical risk class."
    result.attrs["status_semantics"] = "Health-score-derived lot status, separate from composite risk levels."
    return result


def get_health_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return health-score bands, separate from composite-risk levels."""
    categories = ["Critical", "Watch", "Good"]
    if _is_empty(df):
        result = pd.DataFrame(
            {"category": categories, "count": [0, 0, 0], "percentage": [0.0, 0.0, 0.0]}
        )
        result.attrs["category_semantics"] = "Health-score bands, not composite LOW/MEDIUM/HIGH risk levels."
        return result
    _require_columns(df, ["health_score"], "get_health_distribution")

    categories = pd.cut(
        df["health_score"],
        bins=[-np.inf, 50, 75, np.inf],
        labels=["Critical", "Watch", "Good"],
    )

    result = (
        categories.value_counts()
        .reindex(["Critical", "Watch", "Good"], fill_value=0)
        .reset_index()
    )

    result.columns = ["category", "count"]

    result["percentage"] = (
        result["count"] / len(df) * 100
    ).round(1)

    result.attrs["category_semantics"] = "Health-score bands, not composite LOW/MEDIUM/HIGH risk levels."
    return result


def get_health_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average health-related values at each burn-in checkpoint.

    This uses the actual measured values from the dataset rather than
    the old mock trend.
    """

    checkpoints = {
        0: "value_0h",
        24: "value_24h",
        96: "value_96h",
        168: "value_168h",
    }

    if _is_empty(df):
        return pd.DataFrame(columns=["hours", "health_score", "metric_basis"])
    _require_columns(df, list(checkpoints.values()), "get_health_trend")

    rows = []

    baseline = df["value_0h"].mean()

    for hours, column in checkpoints.items():
        mean_value = df[column].mean()

        if baseline == 0:
            health = 100.0
        else:
            health = max(
                0.0,
                min(
                    100.0,
                    100.0 - ((mean_value - baseline) / baseline * 100.0),
                ),
            )

        rows.append(
            {
                "hours": hours,
                "health_score": round(health, 1),
                "metric_basis": "Derived from average measured value relative to the 0h baseline; not a model-generated health score.",
            }
        )

    return pd.DataFrame(rows)


def _require_dataframe(df: pd.DataFrame, context: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{context}: expected a pandas DataFrame.")


def _is_empty(df: pd.DataFrame) -> bool:
    _require_dataframe(df, "dashboard data helper")
    return df.empty


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: analyzed dataframe is missing required column(s): {missing}.")


def _optional_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _rounded_optional_column(df: pd.DataFrame, column: str, decimals: int) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").round(decimals)
    return pd.Series(pd.NA, index=df.index, dtype="Float64")
