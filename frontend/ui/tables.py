"""
tables.py
---------
Renders the "Top 5 High-Risk Components" and "Lot Health Summary"
tables as hand-built HTML with INLINE styles (badge colors, borders,
text colors) so they render correctly even if the external stylesheet
from ui/theme.py hasn't loaded.

Data source: the session controller's already-analyzed DataFrame
(data.session.get_analyzed_dataset()) — this module NEVER runs the ML
pipeline itself and NEVER imports frontend/mock/mock_data.py.
"""

import streamlit as st
import pandas as pd

from data.session import get_analyzed_dataset
from data.dashboard_data import get_high_risk_components, get_lot_health_summary

_TEXT_PRIMARY = "#F3F5F9"
_TEXT_SECONDARY = "#A6B0C3"
_TEXT_MUTED = "#6C7890"
_BORDER = "#232C40"
_BORDER_SOFT = "#1C2436"

# Composite risk levels (LOW/MEDIUM/HIGH) and health-score-based status
# (Critical/Watch/Good) are two DIFFERENT concepts (see
# ml/system_analysis.py) — both palettes live in this one dict, keyed
# by the exact strings each concept actually produces. No new colors
# are introduced: MEDIUM/LOW/title-case "Critical"/"Watch" reuse the
# same hex pairs already defined here for the equivalent severity.
_BADGE_COLORS = {
    # Composite risk levels (ml.system_analysis composite_risk_level)
    "HIGH": ("#3A1F08", "#FB923C"),
    "MEDIUM": ("#3A2B06", "#FBBF24"),
    "LOW": ("#0F2E22", "#4ADE80"),
    # Health-score-based lot status (frontend.data.dashboard_data get_lot_health_summary)
    "Critical": ("#3D1220", "#FB7185"),
    "Watch": ("#3A2B06", "#FBBF24"),
    "Good": ("#0F2E22", "#4ADE80"),
    # Legacy/uppercase variants kept for backward compatibility with
    # any other caller still passing these exact strings.
    "WATCH": ("#3A2B06", "#FBBF24"),
    "CRITICAL": ("#3D1220", "#FB7185"),
    "At Risk": ("#3A2B06", "#FBBF24"),
}

_DEFAULT_BADGE = ("#3A2B06", "#FBBF24")


def _badge(text) -> str:
    label = "N/A" if pd.isna(text) else str(text)
    bg, fg = _BADGE_COLORS.get(label, _DEFAULT_BADGE)
    return (
        f'<span style="display:inline-block; padding:0.18rem 0.6rem; border-radius:6px;'
        f'font-size:0.72rem; font-weight:700; background-color:{bg}; color:{fg};'
        f'white-space:nowrap;">{label}</span>'
    )


def _fmt_number(value, decimals: int = 1) -> str:
    """Format a numeric cell, handling missing/NaN values without raising."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _table_html(headers: list, rows_html: str) -> str:
    head_cells = "".join(
        f'<th style="text-align:left; color:{_TEXT_MUTED}; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.05em; text-transform:uppercase; padding:0.5rem 0.6rem; '
        f'border-bottom:1px solid {_BORDER};">{h}</th>'
        for h in headers
    )
    return (
        f'<table style="width:100%; border-collapse:collapse; font-size:0.86rem; '
        f'table-layout:fixed;">'
        f"<thead><tr>{head_cells}</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _section_header(title: str, button_label: str, button_key: str) -> bool:
    """Renders the section title + a toggle button; returns True if the
    button was just clicked this run (caller decides what that means)."""
    header_col, btn_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; color:{_TEXT_PRIMARY};">'
            f"{title}</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        clicked = st.button(button_label, key=button_key, use_container_width=True)
    return clicked


def _empty_state(message: str) -> None:
    st.markdown(
        f'<div style="padding:1.2rem 0.6rem; color:{_TEXT_MUTED}; font-size:0.85rem; '
        f'border:1px dashed {_BORDER}; border-radius:8px; text-align:center;">{message}</div>',
        unsafe_allow_html=True,
    )


def _get_analyzed_df():
    """Fetch the session controller's analyzed dataset without ever
    triggering ML from this module. Returns None on any failure so
    callers can show a clean empty-state instead of a traceback."""
    try:
        df = get_analyzed_dataset()
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    return df


# ------------------------------------------------------------------
# Lot summary column-contract resolution
# ------------------------------------------------------------------
# get_lot_health_summary()'s exact column names have drifted before
# (e.g. "Avg_Health" vs "Avg Health"), so rather than hard-coding one
# literal string, each logical field is resolved against a list of
# plausible real names actually seen from that helper. This is the
# fix for `KeyError: 'High Risk'` — that column is looked up here
# instead of assumed.
_LOT_ID_CANDIDATES = ["Lot ID", "lot_id"]
_COMPONENTS_CANDIDATES = ["Components", "component_count", "Component Count", "total_components"]
_HIGH_RISK_CANDIDATES = [
    "High Risk", "high_risk", "High_Risk", "high_risk_count",
    "High Risk Count", "composite_high_risk_count", "HighRisk",
]
_AVG_HEALTH_CANDIDATES = [
    "Average Health", "Avg Health", "avg_health", "Avg_Health",
    "average_health_score", "avg_health_score",
]
_STATUS_CANDIDATES = ["Status", "status", "lot_status"]


def _resolve_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_lot_summary(lot_summary, analyzed_df):
    """
    Return a copy of get_lot_health_summary()'s output with fixed,
    canonical display columns: Lot ID / Components / High Risk /
    Average Health / Status — resolving whatever real column names
    the helper actually returns.

    "High Risk" specifically means the composite-risk HIGH count for
    that lot (never the health-status "Critical" count — see the
    module docstring's badge-color note for why those stay separate).
    If get_lot_health_summary() doesn't provide that count under any
    known name, it is derived directly from the analyzed DataFrame
    using the SAME composite_risk_level == "HIGH" definition used
    everywhere else in AnomeX (ml/system_analysis.py) — never a new
    or invented risk definition.
    """
    working = lot_summary.copy()

    lot_id_col = _resolve_column(working, _LOT_ID_CANDIDATES)
    components_col = _resolve_column(working, _COMPONENTS_CANDIDATES)
    high_risk_col = _resolve_column(working, _HIGH_RISK_CANDIDATES)
    avg_health_col = _resolve_column(working, _AVG_HEALTH_CANDIDATES)
    status_col = _resolve_column(working, _STATUS_CANDIDATES)

    if (
        high_risk_col is None
        and lot_id_col is not None
        and analyzed_df is not None
        and "lot_id" in analyzed_df.columns
        and "composite_risk_level" in analyzed_df.columns
    ):
        derived = (
            analyzed_df[analyzed_df["composite_risk_level"] == "HIGH"]
            .groupby("lot_id")
            .size()
            .rename("__derived_high_risk__")
            .reset_index()
        )
        working = working.merge(derived, left_on=lot_id_col, right_on="lot_id", how="left")
        working["__derived_high_risk__"] = working["__derived_high_risk__"].fillna(0).astype(int)
        high_risk_col = "__derived_high_risk__"

    result = pd.DataFrame(index=working.index)
    result["Lot ID"] = working[lot_id_col] if lot_id_col else "N/A"
    result["Components"] = working[components_col] if components_col else None
    result["High Risk"] = working[high_risk_col] if high_risk_col else None
    result["Average Health"] = working[avg_health_col] if avg_health_col else None
    result["Status"] = working[status_col] if status_col else "N/A"
    return result.reset_index(drop=True)


# ============================================================
# High-Risk Components
# ============================================================
def render_high_risk_table() -> None:
    view_all_clicked = _section_header(
        "Top 5 High-Risk Components", "View All", "view_all_high_risk_btn"
    )
    if view_all_clicked:
        st.session_state["show_all_components"] = not st.session_state.get(
            "show_all_components", False
        )

    df = _get_analyzed_df()
    if df is None:
        _empty_state("No analyzed dataset yet. Load or analyze a dataset to see high-risk components.")
        return

    if "composite_risk_level" not in df.columns or "composite_risk_score" not in df.columns:
        _empty_state("Composite risk data is not available for this dataset yet.")
        return

    high_risk_df = df[df["composite_risk_level"] == "HIGH"]

    if high_risk_df.empty:
        st.markdown(
            f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin-bottom:0.5rem;">'
            f"Showing components with highest risk levels</div>",
            unsafe_allow_html=True,
        )
        _empty_state("No components are currently classified as HIGH composite risk.")
    else:
        top5 = get_high_risk_components(high_risk_df, limit=5)
        rows_html = ""
        td_style = f'padding:0.6rem 0.6rem; border-bottom:1px solid {_BORDER_SOFT}; color:{_TEXT_PRIMARY};'
        for _, r in top5.iterrows():
            rows_html += f"""
            <tr>
                <td style="{td_style} font-weight:700;">{r['Component ID']}</td>
                <td style="{td_style} color:{_TEXT_SECONDARY};">{r['Lot ID']}</td>
                <td style="{td_style}">{_fmt_number(r['Health Score'])}</td>
                <td style="{td_style}">{_fmt_number(r['Anomaly Score'])}</td>
                <td style="{td_style}">{_fmt_number(r['Predicted 168h'])}</td>
                <td style="{td_style}">{_badge(r['Risk Level'])}</td>
            </tr>
            """

        st.html(
            _table_html(
                ["Component ID", "Lot ID", "Health Score", "Anomaly Score", "Predicted 168h", "Risk Level"],
                rows_html,
            )
        )
        st.markdown(
            f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin-top:0.5rem;">'
            f"Showing components with highest risk levels &middot; Anomaly Score is a "
            f"normalized 0-100 score</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.get("show_all_components", False):
        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        _render_all_components(df)


def _render_all_components(df) -> None:
    st.markdown(
        f'<div style="font-size:0.95rem; font-weight:700; color:{_TEXT_PRIMARY}; '
        f'border-top:1px solid {_BORDER}; padding-top:1rem;">All Components</div>',
        unsafe_allow_html=True,
    )

    search_col, lot_col, risk_col, close_col = st.columns([2, 1, 1, 1])
    with search_col:
        search_term = st.text_input(
            "Search Component ID", value="", key="all_components_search", placeholder="e.g. C00054"
        )
    with lot_col:
        lot_options = ["All"] + sorted(df["lot_id"].dropna().astype(str).unique().tolist()) if "lot_id" in df.columns else ["All"]
        lot_filter = st.selectbox("Lot ID", lot_options, key="all_components_lot_filter")
    with risk_col:
        risk_options = ["All", "HIGH", "MEDIUM", "LOW"]
        risk_filter = st.selectbox("Risk Level", risk_options, key="all_components_risk_filter")
    with close_col:
        st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("Close", key="close_all_components", use_container_width=True):
            st.session_state["show_all_components"] = False
            st.rerun()

    filtered = df.copy()
    if search_term and "component_id" in filtered.columns:
        filtered = filtered[filtered["component_id"].astype(str).str.contains(search_term, case=False, na=False)]
    if lot_filter != "All" and "lot_id" in filtered.columns:
        filtered = filtered[filtered["lot_id"].astype(str) == lot_filter]
    if risk_filter != "All" and "composite_risk_level" in filtered.columns:
        filtered = filtered[filtered["composite_risk_level"] == risk_filter]

    sort_options = {
        "Composite Risk Score": "composite_risk_score",
        "Health Score": "health_score",
        "Anomaly Score": "anomaly_score",
        "Predicted 168h": "predicted_168h",
    }
    available_sorts = {label: col for label, col in sort_options.items() if col in filtered.columns}

    if available_sorts:
        sort_col1, sort_col2 = st.columns([2, 1])
        with sort_col1:
            sort_label = st.selectbox("Sort by", list(available_sorts.keys()), key="all_components_sort_by")
        with sort_col2:
            descending = st.checkbox("Descending", value=True, key="all_components_sort_desc")
        filtered = filtered.sort_values(available_sorts[sort_label], ascending=not descending)

    st.markdown(
        f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin:0.4rem 0;">'
        f"{len(filtered)} of {len(df)} components</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        _empty_state("No components match the current search/filter.")
        return

    display_columns = {
        "component_id": "Component ID",
        "lot_id": "Lot ID",
        "health_score": "Health Score",
        "anomaly_score": "Anomaly Score",
        "predicted_168h": "Predicted 168h",
        "composite_risk_score": "Composite Risk Score",
        "composite_risk_level": "Risk Level",
    }
    available_display_columns = [c for c in display_columns if c in filtered.columns]
    display_df = filtered[available_display_columns].rename(columns=display_columns)

    st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)

    if "component_id" in filtered.columns and not filtered.empty:
        inspect_id = st.selectbox(
            "Inspect a component",
            filtered["component_id"].astype(str).tolist(),
            key="all_components_inspect",
        )
        if inspect_id:
            _render_component_detail(filtered[filtered["component_id"].astype(str) == inspect_id].iloc[0])


def _render_component_detail(row) -> None:
    fields = [
        ("Health Score", row.get("health_score")),
        ("Anomaly Score", row.get("anomaly_score")),
        ("Failure Probability", row.get("failure_probability")),
        ("Predicted 168h", row.get("predicted_168h")),
        ("Composite Risk Score", row.get("composite_risk_score")),
        ("Composite Risk Level", row.get("composite_risk_level")),
        ("Value 0h", row.get("value_0h")),
        ("Value 24h", row.get("value_24h")),
        ("Value 96h", row.get("value_96h")),
        ("Value 168h", row.get("value_168h")),
    ]
    available_fields = [(label, value) for label, value in fields if value is not None and not (isinstance(value, float) and pd.isna(value))]
    if not available_fields:
        return

    cells = ""
    for label, value in available_fields:
        display_value = _badge(value) if label == "Composite Risk Level" else _fmt_number(value) if isinstance(value, (int, float)) else str(value)
        cells += (
            f'<div style="padding:0.5rem 0.7rem; border:1px solid {_BORDER}; border-radius:8px;">'
            f'<div style="font-size:0.68rem; color:{_TEXT_MUTED}; text-transform:uppercase; '
            f'letter-spacing:0.04em;">{label}</div>'
            f'<div style="font-size:0.95rem; color:{_TEXT_PRIMARY}; font-weight:600; margin-top:0.15rem;">'
            f'{display_value}</div></div>'
        )

    st.markdown(
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); '
        f'gap:0.6rem; margin-top:0.6rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Lot Health Summary
# ============================================================
def render_lot_summary_table() -> None:
    view_all_clicked = _section_header("Lot Health Summary", "View All", "view_all_lots_btn")
    if view_all_clicked:
        st.session_state["show_all_lots"] = not st.session_state.get("show_all_lots", False)

    df = _get_analyzed_df()
    if df is None:
        _empty_state("No analyzed dataset yet. Load or analyze a dataset to see lot health.")
        return

    try:
        lot_summary_raw = get_lot_health_summary(df)
    except Exception:
        _empty_state("Lot summary is not available for this dataset yet.")
        return

    if lot_summary_raw is None or lot_summary_raw.empty:
        _empty_state("No lot data available.")
    else:
        lot_summary = _normalize_lot_summary(lot_summary_raw, df)
        compact = lot_summary.head(5)
        rows_html = ""
        td_style = f'padding:0.6rem 0.6rem; border-bottom:1px solid {_BORDER_SOFT}; color:{_TEXT_PRIMARY};'
        for _, r in compact.iterrows():
            rows_html += f"""
            <tr>
                <td style="{td_style} font-weight:700;">{r['Lot ID']}</td>
                <td style="{td_style} color:{_TEXT_SECONDARY};">{_fmt_number(r['Components'], 0)}</td>
                <td style="{td_style}">{_fmt_number(r['High Risk'], 0)}</td>
                <td style="{td_style}">{_fmt_number(r['Average Health'])}</td>
                <td style="{td_style}">{_badge(r['Status'])}</td>
            </tr>
            """

        st.html(
            _table_html(
                ["Lot ID", "Components", "High Risk", "Average Health", "Status"],
                rows_html,
            )
        )
        st.markdown(
            f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin-top:0.5rem;">'
            f"Lot-level reliability summary &middot; Status reflects health score, not composite risk level"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.get("show_all_lots", False):
        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        _render_all_lots(df)


def _render_all_lots(df) -> None:
    try:
        lot_summary_raw = get_lot_health_summary(df)
    except Exception:
        _empty_state("Lot summary is not available for this dataset yet.")
        return

    st.markdown(
        f'<div style="font-size:0.95rem; font-weight:700; color:{_TEXT_PRIMARY}; '
        f'border-top:1px solid {_BORDER}; padding-top:1rem;">All Lots</div>',
        unsafe_allow_html=True,
    )

    if lot_summary_raw is None or lot_summary_raw.empty:
        _empty_state("No lot data available.")
        return

    lot_summary = _normalize_lot_summary(lot_summary_raw, df)

    search_col, sort_col, close_col = st.columns([2, 2, 1])
    with search_col:
        search_term = st.text_input("Search Lot ID", value="", key="all_lots_search", placeholder="e.g. L1")
    with sort_col:
        sort_options = [c for c in ["Components", "High Risk", "Average Health"] if c in lot_summary.columns]
        sort_label = st.selectbox("Sort by", sort_options, key="all_lots_sort_by") if sort_options else None
    with close_col:
        st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("Close", key="close_all_lots", use_container_width=True):
            st.session_state["show_all_lots"] = False
            st.rerun()

    filtered = lot_summary.copy()
    if search_term:
        filtered = filtered[filtered["Lot ID"].astype(str).str.contains(search_term, case=False, na=False)]
    if sort_label:
        filtered = filtered.sort_values(sort_label, ascending=False)

    st.markdown(
        f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin:0.4rem 0;">'
        f"{len(filtered)} of {len(lot_summary)} lots</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        _empty_state("No lots match the current search.")
        return

    st.dataframe(filtered, use_container_width=True, height=380, hide_index=True)