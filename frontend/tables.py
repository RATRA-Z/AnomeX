"""
tables.py
---------
Renders the "Top 5 High-Risk Components" and "Lot Health Summary"
tables as hand-built HTML with INLINE styles (badge colors, borders,
text colors) so they render correctly even if the external stylesheet
from ui/theme.py hasn't loaded.
"""

import streamlit as st

from mock.mock_data import HIGH_RISK_COMPONENTS, LOT_HEALTH_SUMMARY

_TEXT_PRIMARY = "#F3F5F9"
_TEXT_SECONDARY = "#A6B0C3"
_TEXT_MUTED = "#6C7890"
_BORDER = "#232C40"
_BORDER_SOFT = "#1C2436"

_BADGE_COLORS = {
    "HIGH": ("#3A1F08", "#FB923C"),
    "WATCH": ("#3A2B06", "#FBBF24"),
    "CRITICAL": ("#3D1220", "#FB7185"),
    "Good": ("#0F2E22", "#4ADE80"),
    "At Risk": ("#3A2B06", "#FBBF24"),
}


def _badge(text: str) -> str:
    bg, fg = _BADGE_COLORS.get(text, ("#3A2B06", "#FBBF24"))
    return (
        f'<span style="display:inline-block; padding:0.18rem 0.6rem; border-radius:6px;'
        f'font-size:0.72rem; font-weight:700; background-color:{bg}; color:{fg};'
        f'white-space:nowrap;">{text}</span>'
    )


def _table_html(headers: list, rows_html: str) -> str:
    head_cells = "".join(
        f'<th style="text-align:left; color:{_TEXT_MUTED}; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.05em; text-transform:uppercase; padding:0.5rem 0.6rem; '
        f'border-bottom:1px solid {_BORDER};">{h}</th>'
        for h in headers
    )
    return (
        f'<table style="width:100%; border-collapse:collapse; font-size:0.86rem;">'
        f"<thead><tr>{head_cells}</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def render_high_risk_table() -> None:
    header_col, btn_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; color:{_TEXT_PRIMARY};">'
            f"Top 5 High-Risk Components</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        st.button("View All", key="view_all_high_risk", use_container_width=True)

    df = HIGH_RISK_COMPONENTS
    rows_html = ""
    td_style = f'padding:0.6rem 0.6rem; border-bottom:1px solid {_BORDER_SOFT}; color:{_TEXT_PRIMARY};'
    for _, r in df.iterrows():
        rows_html += f"""
        <tr>
            <td style="{td_style} font-weight:700;">{r['Component ID']}</td>
            <td style="{td_style} color:{_TEXT_SECONDARY};">{r['Lot ID']}</td>
            <td style="{td_style}">{r['Health Score']}</td>
            <td style="{td_style}">{r['Anomaly Score']}%</td>
            <td style="{td_style}">{r['Predicted 168h']:.1f} \u00b5A</td>
            <td style="{td_style}">{_badge(r['Risk Level'])}</td>
        </tr>
        """

    st.markdown(
        _table_html(
            ["Component ID", "Lot ID", "Health Score", "Anomaly Score", "Predicted 168h", "Risk Level"],
            rows_html,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin-top:0.5rem;">'
        f"Showing components with highest risk levels</div>",
        unsafe_allow_html=True,
    )


def render_lot_summary_table() -> None:
    header_col, btn_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; color:{_TEXT_PRIMARY};">'
            f"Lot Health Summary</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        st.button("View All", key="view_all_lots", use_container_width=True)

    df = LOT_HEALTH_SUMMARY.head(5)
    rows_html = ""
    td_style = f'padding:0.6rem 0.6rem; border-bottom:1px solid {_BORDER_SOFT}; color:{_TEXT_PRIMARY};'
    for _, r in df.iterrows():
        rows_html += f"""
        <tr>
            <td style="{td_style} font-weight:700;">{r['Lot ID']}</td>
            <td style="{td_style} color:{_TEXT_SECONDARY};">{r['Components']}</td>
            <td style="{td_style}">{r['High Risk']}</td>
            <td style="{td_style}">{r['Critical']}</td>
            <td style="{td_style}">{r['Avg Health']}</td>
            <td style="{td_style}">{_badge(r['Status'])}</td>
        </tr>
        """

    st.markdown(
        _table_html(
            ["Lot ID", "Components", "High Risk", "Critical", "Avg Health", "Status"],
            rows_html,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin-top:0.5rem;">'
        f"Lot-level reliability summary and risk status</div>",
        unsafe_allow_html=True,
    )