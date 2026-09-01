"""
cards.py
--------
Small reusable presentational components: the top header row (title +
system status + action buttons), the four KPI cards, and the bottom
CTA banner.

All numbers rendered here come from mock.mock_data.DASHBOARD_METRICS --
nothing is typed in locally.

NOTE: the header uses a single FLAT row of st.columns() (not columns
nested inside columns). Nested columns are a known source of unreliable
sizing in Streamlit -- that was causing the Upload/Analyze buttons to
render oversized. The KPI cards below also use INLINE styles rather
than relying purely on the .kpi-card class, so the card borders/
background always show up even if the external stylesheet fails to
load for any reason.
"""

import streamlit as st

from mock.mock_data import DASHBOARD_METRICS

_TEXT_PRIMARY = "#F3F5F9"
_TEXT_MUTED = "#6C7890"
_BG_CARD = "#121826"
_BORDER = "#232C40"
_NORMAL = "#4ADE80"
_WATCH = "#FBBF24"
_CRITICAL = "#FB7185"
_STATUS_ONLINE_BG = "#0F2E22"


def render_header(on_upload=None, on_analyze=None) -> None:
    # One flat row: title | status pill | upload | analyze
    title_col, status_col, upload_col, analyze_col = st.columns([2.3, 1.3, 1.05, 1.35])

    with title_col:
        st.markdown(
            f'<div style="font-size:1.8rem; font-weight:800; color:{_TEXT_PRIMARY}; margin:0;">'
            f"Dashboard Overview</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:{_TEXT_MUTED}; font-size:0.92rem; margin-top:0.1rem;">'
            f"Real-time reliability insights from burn-in analysis</div>",
            unsafe_allow_html=True,
        )

    with status_col:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:flex-end;
                        height:2.5rem; white-space:nowrap;">
                <span style="display:inline-flex; align-items:center; gap:0.4rem;
                             background-color:{_STATUS_ONLINE_BG}; color:{_NORMAL};
                             padding:0.35rem 0.75rem; border-radius:999px;
                             font-size:0.76rem; font-weight:700;">
                    <span style="width:7px; height:7px; border-radius:50%;
                                 background-color:{_NORMAL}; display:inline-block;"></span>
                    SYSTEM STATUS: ONLINE
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with upload_col:
        if st.button("\u2B06\uFE0F Upload", key="header_upload_btn", use_container_width=True):
            st.session_state["show_uploader"] = True
            if on_upload:
                on_upload()

    with analyze_col:
        if st.button("\u25B6\uFE0F Analyze Dataset", key="header_analyze_btn", use_container_width=True):
            st.session_state["analysis_triggered"] = True
            if on_analyze:
                on_analyze()

    if st.session_state.get("analysis_triggered"):
        st.info(
            "Demo analysis complete on **burn_in_data.csv** — showing placeholder results below. "
            "Real anomaly detection will run here once the backend is connected.",
            icon="\u2705",
        )


def _kpi_card(label: str, value_html: str, sub_html: str, sub_color: str, icon: str) -> str:
    return f"""
    <div style="background-color:{_BG_CARD}; border:1px solid {_BORDER}; border-radius:14px;
                padding:1rem 1.1rem; height:100%;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="color:{_TEXT_MUTED}; font-size:0.7rem; font-weight:700;
                        letter-spacing:0.06em; text-transform:uppercase;">{label}</div>
            <div style="font-size:1.2rem;">{icon}</div>
        </div>
        <div style="color:{_TEXT_PRIMARY}; font-size:2.05rem; font-weight:800;
                    line-height:1.15; margin-top:0.15rem;">{value_html}</div>
        <div style="color:{sub_color}; font-size:0.78rem; margin-top:0.15rem;">{sub_html}</div>
    </div>
    """


def render_kpi_row() -> None:
    m = DASHBOARD_METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _kpi_card(
                "TOTAL COMPONENTS",
                f"{m['total_components']:,}",
                f"{m['total_components_pct']:.0f}% of dataset",
                _TEXT_MUTED,
                "\U0001F9E9",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _kpi_card(
                "HIGH RISK",
                f"{m['high_risk']:,}",
                f"{m['high_risk_pct']}% of total",
                _WATCH,
                "\u26A0\uFE0F",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _kpi_card(
                "CRITICAL",
                f"{m['critical']:,}",
                f"{m['critical_pct']}% of total",
                _CRITICAL,
                "\U0001F6A8",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _kpi_card(
                "SYSTEM HEALTH",
                f'{m["system_health"]}<span style="color:{_TEXT_MUTED}; font-size:1.1rem; '
                f'font-weight:600;"> /{m["system_health_max"]}</span>',
                "Overall Reliability Score",
                _NORMAL,
                "\U0001F49A",
            ),
            unsafe_allow_html=True,
        )


def render_cta_banner() -> None:
    st.markdown(
        f"""
        <div style="margin-top:1rem; background:linear-gradient(90deg, #1E1B3A 0%, {_BG_CARD} 100%);
                    border:1px solid {_BORDER}; border-radius:16px; padding:1.2rem 1.4rem;">
            <div style="display:flex; align-items:flex-start; gap:0.9rem;">
                <div style="font-size:1.5rem;">\u2728</div>
                <div>
                    <div style="color:{_TEXT_PRIMARY}; font-size:1.1rem; font-weight:800;
                                margin-bottom:0.15rem;">Start Your Analysis</div>
                    <div style="color:#A6B0C3; font-size:0.85rem;">
                        Navigate to component-level analysis or explore system-level
                        reliability insights.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, btn_col = st.columns([4, 1.3])
    with btn_col:
        if st.button("Go to Components \u2192", key="cta_go_components", use_container_width=True):
            st.session_state["active_page"] = "Components"
            st.rerun()