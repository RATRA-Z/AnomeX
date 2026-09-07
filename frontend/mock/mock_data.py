"""
mock_data.py
------------
Centralized PLACEHOLDER / DEMO data source for the AnomeX frontend.

Nothing in the UI layer should hardcode a number. Every KPI, chart, and
table on the Overview page reads from the structures defined here, so
that:

  1. All displayed numbers are guaranteed to be internally consistent
     (KPI cards == donut chart == table summaries).
  2. Later, a backend engineer can swap the contents of this file for
     real Pandas DataFrames / ML outputs without touching any UI code.

Future architecture:

    CSV / Database
          |
    Pandas DataFrame
          |
    Python Backend / ML
          |
    Processed Results   <-- this file will eventually be generated here
          |
    AnomeX Frontend (ui/*.py, pages/*.py)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. DATASET METADATA (shown at the bottom of the sidebar)
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "file_name": "burn_in_data.csv",
    "rows": 10_000,
    "components": 10_000,
    "lots": 20,
    "primary_parameter": "Leakage Current (\u00b5A)",
}

# ---------------------------------------------------------------------------
# 2. SINGLE SOURCE OF TRUTH: COMPONENT HEALTH COUNTS
# ---------------------------------------------------------------------------
# Every other number on the page (KPI cards, donut chart, percentages,
# lot table roll-ups) is derived from these four counts so nothing can
# ever go out of sync.

TOTAL_COMPONENTS = DATASET_INFO["components"]

HEALTH_COUNTS = {
    "Critical": 30,
    "High Risk": 182,
    "Watch": 868,
    # Normal is whatever remains -- never typed in by hand.
}
HEALTH_COUNTS["Normal"] = TOTAL_COMPONENTS - sum(HEALTH_COUNTS.values())

# Sanity check kept on purpose: if someone edits the counts above without
# updating Normal correctly, fail loudly instead of silently drifting.
assert sum(HEALTH_COUNTS.values()) == TOTAL_COMPONENTS, (
    "Health category counts must sum to TOTAL_COMPONENTS."
)

# Display order used everywhere (donut legend, color mapping, etc.)
HEALTH_CATEGORY_ORDER = ["Normal", "Watch", "High Risk", "Critical"]

HEALTH_COLORS = {
    "Normal": "#22C55E",     # green
    "Watch": "#F5B700",      # amber
    "High Risk": "#FF8A3D",  # orange
    "Critical": "#F0475C",   # red
}


def get_health_distribution() -> pd.DataFrame:
    """Return the health-distribution table with counts + calculated %.

    Percentages are always computed here (count / total * 100), never
    typed in as literals, so they can never disagree with the counts.
    """
    df = pd.DataFrame(
        {
            "category": HEALTH_CATEGORY_ORDER,
            "count": [HEALTH_COUNTS[c] for c in HEALTH_CATEGORY_ORDER],
        }
    )
    df["percentage"] = (df["count"] / TOTAL_COMPONENTS * 100).round(1)
    return df


# ---------------------------------------------------------------------------
# 3. TOP-LEVEL KPI METRICS
# ---------------------------------------------------------------------------
# System Health is the one figure that is a genuine composite "score"
# rather than a plain count, so it's modeled as a placeholder weighted
# score. It still lives in this single dict so the UI never hardcodes it.

_SYSTEM_HEALTH_WEIGHTS = {"Normal": 100, "Watch": 70, "High Risk": 40, "Critical": 10}
_weighted_sum = sum(
    HEALTH_COUNTS[cat] * score for cat, score in _SYSTEM_HEALTH_WEIGHTS.items()
)
SYSTEM_HEALTH_SCORE = round(_weighted_sum / TOTAL_COMPONENTS)

DASHBOARD_METRICS = {
    "total_components": TOTAL_COMPONENTS,
    "total_components_pct": 100.0,
    "high_risk": HEALTH_COUNTS["High Risk"],
    "high_risk_pct": round(HEALTH_COUNTS["High Risk"] / TOTAL_COMPONENTS * 100, 2),
    "critical": HEALTH_COUNTS["Critical"],
    "critical_pct": round(HEALTH_COUNTS["Critical"] / TOTAL_COMPONENTS * 100, 2),
    "system_health": SYSTEM_HEALTH_SCORE,
    "system_health_max": 100,
}

# ---------------------------------------------------------------------------
# 4. HEALTH SCORE TREND (burn-in time series)
# ---------------------------------------------------------------------------
# Stored as (hours, score) pairs so numeric ordering is preserved -- the
# chart must treat the x-axis as elapsed hours, not category labels.

HEALTH_TREND_HOURS = [0, 24, 96, 168]

HEALTH_TREND_SERIES = {
    "Overall": [90, 82, 76, 74],
    "Lot": [92, 84, 78, 75],
    "Component Type": [88, 80, 73, 70],
    "Manufacturer": [91, 83, 77, 76],
}

HEALTH_TREND_OPTIONS = list(HEALTH_TREND_SERIES.keys())


def get_health_trend(view: str = "Overall") -> pd.DataFrame:
    """Return a chronologically ordered health-score trend for a view."""
    series = HEALTH_TREND_SERIES.get(view, HEALTH_TREND_SERIES["Overall"])
    return pd.DataFrame(
        {
            "hours": HEALTH_TREND_HOURS,
            "label": [f"{h}h" for h in HEALTH_TREND_HOURS],
            "health_score": series,
        }
    ).sort_values("hours")


# ---------------------------------------------------------------------------
# 5. TOP HIGH-RISK COMPONENTS TABLE
# ---------------------------------------------------------------------------

HIGH_RISK_COMPONENTS = pd.DataFrame(
    [
        {"Component ID": "C0184", "Lot ID": "LOT_17", "Health Score": 28,
         "Anomaly Score": 94, "Predicted 168h": 54.8, "Risk Level": "HIGH"},
        {"Component ID": "C0219", "Lot ID": "LOT_17", "Health Score": 34,
         "Anomaly Score": 88, "Predicted 168h": 51.2, "Risk Level": "HIGH"},
        {"Component ID": "C0301", "Lot ID": "LOT_08", "Health Score": 61,
         "Anomaly Score": 63, "Predicted 168h": 37.4, "Risk Level": "WATCH"},
        {"Component ID": "C0442", "Lot ID": "LOT_12", "Health Score": 58,
         "Anomaly Score": 61, "Predicted 168h": 36.1, "Risk Level": "WATCH"},
        {"Component ID": "C0520", "Lot ID": "LOT_03", "Health Score": 55,
         "Anomaly Score": 59, "Predicted 168h": 35.6, "Risk Level": "WATCH"},
    ]
)
# These are DEMO/placeholder values only -- not real ML predictions.


# ---------------------------------------------------------------------------
# 6. LOT HEALTH SUMMARY TABLE
# ---------------------------------------------------------------------------
# Generated (not hand-typed) so that, per-lot:
#   - components across all lots sum to TOTAL_COMPONENTS
#   - high_risk + critical never exceeds components for that lot
#   - the sum of per-lot high_risk / critical equals the overall KPI totals
#   - avg_health stays within [0, 100]
#   - status is derived FROM the displayed health/risk numbers

def _build_lot_summary() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)

    n_lots = DATASET_INFO["lots"]
    lot_ids = [f"LOT_{str(i).zfill(2)}" for i in range(1, n_lots + 1)]
    components_per_lot = TOTAL_COMPONENTS // n_lots  # 500 each, divides evenly

    # Distribute the overall high-risk / critical totals across lots using a
    # multinomial draw so the per-lot numbers are irregular but always sum
    # back exactly to the KPI totals (182 / 30).
    high_risk_weights = rng.dirichlet(np.ones(n_lots) * 1.5)
    critical_weights = rng.dirichlet(np.ones(n_lots) * 1.2)

    high_risk_per_lot = rng.multinomial(HEALTH_COUNTS["High Risk"], high_risk_weights)
    critical_per_lot = rng.multinomial(HEALTH_COUNTS["Critical"], critical_weights)

    # Guard against (rare) rounding cases where high_risk+critical would
    # exceed the lot's component count.
    high_risk_per_lot = np.minimum(high_risk_per_lot, components_per_lot - critical_per_lot - 1)
    high_risk_per_lot = np.clip(high_risk_per_lot, 0, None)

    rows = []
    for i, lot_id in enumerate(lot_ids):
        hr = int(high_risk_per_lot[i])
        cr = int(critical_per_lot[i])
        healthy_fraction = 1 - (hr + cr) / components_per_lot
        # Avg health scaled off how "clean" the lot is, kept within [0, 100]
        avg_health = int(round(np.clip(60 + healthy_fraction * 38, 0, 100)))

        if cr >= 5 or avg_health < 65:
            status = "Critical"
        elif hr >= 15 or avg_health < 75:
            status = "At Risk"
        else:
            status = "Good"

        rows.append(
            {
                "Lot ID": lot_id,
                "Components": components_per_lot,
                "High Risk": hr,
                "Critical": cr,
                "Avg Health": avg_health,
                "Status": status,
            }
        )

    return pd.DataFrame(rows)


LOT_HEALTH_SUMMARY = _build_lot_summary()

# Cross-check: per-lot totals must reconcile with the top-level KPIs.
assert LOT_HEALTH_SUMMARY["Components"].sum() == TOTAL_COMPONENTS
assert LOT_HEALTH_SUMMARY["High Risk"].sum() == HEALTH_COUNTS["High Risk"]
assert LOT_HEALTH_SUMMARY["Critical"].sum() == HEALTH_COUNTS["Critical"]
