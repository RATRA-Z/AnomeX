"""
cards.py
--------
Small reusable presentational components: the top header row (title +
system status + action buttons), the four KPI cards, and the bottom
CTA banner.

KPI values come from the analyzed dataframe retained by ``data.session``.
This module renders state; it does not run ML directly.

NOTE: the header uses a single FLAT row of st.columns() (not columns
nested inside columns). Nested columns are a known source of unreliable
sizing in Streamlit -- that was causing the Upload/Analyze buttons to
render oversized. The KPI cards below also use INLINE styles rather
than relying purely on the .kpi-card class, so the card borders/
background always show up even if the external stylesheet fails to
load for any reason.

KPI CARD LAYOUT NOTE:
Each card is a flex column (label+icon row / value / subtitle) with a
shared min-height and `justify-content: space-between`. Streamlit
stretches columns in a row to the same height automatically, so once
one card is tallest (e.g. its label or subtitle wraps to two lines),
the others stretch to match -- `space-between` then pushes each card's
subtitle to the bottom instead of leaving a visible gap only under the
shorter cards. `min-width: 0` on the label wrapper is required for the
label to actually wrap inside the flex row instead of overflowing.
"""

import streamlit as st

from data.dashboard_data import get_health_distribution, get_kpi_metrics
from data.session import (
    ANALYSIS_ERROR_KEY,
    ANALYSIS_READY_KEY,
    DATASET_SOURCE_KEY,
    accept_uploaded_dataset,
    get_analyzed_dataset,
    get_raw_dataset,
    get_validation_result,
    run_analysis,
)

_TEXT_PRIMARY = "#F3F5F9"
_TEXT_MUTED = "#6C7890"
_BG_CARD = "#121826"
_BORDER = "#232C40"
_NORMAL = "#4ADE80"
_WATCH = "#FBBF24"
_CRITICAL = "#FB7185"
_STATUS_ONLINE_BG = "#0F2E22"
_STATUS_PENDING_BG = "#3A2B06"
_STATUS_ERROR_BG = "#3D1220"

# Shared sizing tokens for the KPI row so all four cards read as one
# consistent row rather than four independently-styled boxes.
_KPI_MIN_HEIGHT = "150px"
_KPI_PADDING = "1rem 1.1rem"


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

    validation_result = get_validation_result()
    source = st.session_state.get(DATASET_SOURCE_KEY)
    source_label = "UPLOADED" if source == "uploaded" else "DEFAULT"
    analysis_error = st.session_state.get(ANALYSIS_ERROR_KEY)
    if analysis_error:
        status_text, status_color, status_background = "ANALYSIS FAILED", _CRITICAL, _STATUS_ERROR_BG
    elif validation_result is not None and not validation_result.is_valid:
        status_text, status_color, status_background = "VALIDATION FAILED", _CRITICAL, _STATUS_ERROR_BG
    elif st.session_state.get(ANALYSIS_READY_KEY, False):
        status_text, status_color, status_background = "ANALYSIS READY", _NORMAL, _STATUS_ONLINE_BG
    else:
        status_text, status_color, status_background = "READY FOR ANALYSIS", _WATCH, _STATUS_PENDING_BG

    with status_col:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:flex-end;
                        height:2.5rem; white-space:nowrap;">
                <span style="display:inline-flex; align-items:center; gap:0.4rem;
                             background-color:{status_background}; color:{status_color};
                             padding:0.35rem 0.75rem; border-radius:999px;
                             font-size:0.76rem; font-weight:700;">
                    <span style="width:7px; height:7px; border-radius:50%;
                                 background-color:{status_color}; display:inline-block;"></span>
                    {source_label}: {status_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with upload_col:
        if st.button("\u2B06\uFE0F Upload", key="header_upload_btn", use_container_width=True):
            # Header uploader gets its own flag rather than reusing the
            # sidebar's `show_uploader` -- sidebar.py renders its own
            # file_uploader unconditionally whenever `show_uploader` is
            # True, so setting that flag here would pop open BOTH
            # uploaders at once. Explicitly clearing `show_uploader` too
            # covers the (unlikely but possible) case where the sidebar
            # uploader was already open when this button is clicked.
            st.session_state["header_show_uploader"] = True
            st.session_state["show_uploader"] = False
            if on_upload:
                on_upload()
            st.rerun()

    with analyze_col:
        if st.button("\u25B6\uFE0F Analyze Dataset", key="header_analyze_btn", use_container_width=True):
            was_ready = bool(st.session_state.get(ANALYSIS_READY_KEY, False))
            with st.spinner("Running AnomeX analysis..."):
                analyzed = run_analysis()
            if analyzed is None:
                st.error(st.session_state.get(ANALYSIS_ERROR_KEY, "Analysis could not be completed."))
            elif was_ready:
                st.info("Current analysis results are already up to date.")
            else:
                st.success(f"Analysis complete for {len(analyzed):,} components.")
            if on_analyze:
                on_analyze()

    if validation_result is not None and not validation_result.is_valid:
        st.warning("Current upload did not pass validation. Review the sidebar details before analyzing.")
    elif not st.session_state.get(ANALYSIS_READY_KEY, False) and get_raw_dataset() is not None:
        st.caption("Dataset is loaded and ready for explicit analysis.")

    # Only show the header's own inline uploader when the sidebar's
    # uploader (`show_uploader`) isn't the one currently open -- this is
    # the coordination point that keeps exactly one upload workflow
    # visible at a time without needing any change to sidebar.py.
    if st.session_state.get("header_show_uploader", False) and not st.session_state.get(
        "show_uploader", False
    ):
        st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)
        _render_inline_uploader()


def _render_inline_uploader() -> None:
    """CSV upload panel shown on the main Overview page below the header row.

    Delegates entirely to the existing ``data.session`` controller
    (``accept_uploaded_dataset`` / ``get_validation_result``) -- this
    function only renders the widget and the controller's own result,
    it never re-implements CSV parsing or column validation itself.
    """
    # Streamlit widgets cannot be reliably nested inside a raw HTML <div>,
    # so use a bordered Streamlit container for the actual uploader card.
    with st.container(border=True):
        st.markdown(
            f'<div style="color:{_TEXT_PRIMARY}; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">'
            f"Upload Dataset</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Select a CSV, then load it for validation. "
            "Analysis still requires an explicit Analyze Dataset step."
        )

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="header_csv_uploader",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.caption(f"Selected: {uploaded.name}")
            action_col, cancel_col = st.columns([1, 1])
            with action_col:
                load_clicked = st.button(
                    "Load Uploaded Dataset",
                    key="header_load_uploaded_dataset",
                    use_container_width=True,
                )
            with cancel_col:
                if st.button(
                    "Cancel",
                    key="header_cancel_uploader",
                    use_container_width=True,
                ):
                    st.session_state["header_show_uploader"] = False
                    st.rerun()

            if load_clicked:
                result = accept_uploaded_dataset(uploaded, filename=uploaded.name)
                if result.is_valid:
                    st.session_state["header_show_uploader"] = False
                    st.success(
                        f"Dataset loaded: {uploaded.name} — "
                        f"{result.row_count:,} rows. Ready for analysis."
                    )
                    st.rerun()
                else:
                    _render_header_validation_feedback(result)
        else:
            if st.button(
                "Cancel",
                key="header_cancel_uploader_empty",
                use_container_width=True,
            ):
                st.session_state["header_show_uploader"] = False
                st.rerun()

        existing_result = get_validation_result()
        if existing_result is not None and not existing_result.is_valid:
            _render_header_validation_feedback(existing_result)



def _render_header_validation_feedback(result) -> None:
    """Display an already-computed validation result. Renders only --
    the validation itself happened inside accept_uploaded_dataset()."""
    for error in result.errors:
        st.error(error)
    if getattr(result, "missing_columns", None):
        st.caption("Missing columns: " + ", ".join(result.missing_columns))
    for column, reasons in getattr(result, "invalid_columns", {}).items():
        st.caption(f"{column}: " + "; ".join(reasons))


def _kpi_card(label: str, value_html: str, sub_html: str, sub_color: str, icon: str) -> str:
    return f"""
    <div style="background-color:{_BG_CARD}; border:1px solid {_BORDER}; border-radius:14px;
                padding:{_KPI_PADDING}; min-height:{_KPI_MIN_HEIGHT}; height:100%;
                box-sizing:border-box; display:flex; flex-direction:column;
                justify-content:space-between; overflow:hidden;">
        <div>
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem;">
                <div style="color:{_TEXT_MUTED}; font-size:0.7rem; font-weight:700;
                            letter-spacing:0.06em; text-transform:uppercase; min-width:0;
                            overflow-wrap:break-word; line-height:1.35;">{label}</div>
                <div style="font-size:1.2rem; line-height:1; flex-shrink:0;">{icon}</div>
            </div>
            <div style="color:{_TEXT_PRIMARY}; font-size:2.05rem; font-weight:800;
                        line-height:1.15; margin-top:0.35rem; overflow-wrap:break-word;">{value_html}</div>
        </div>
        <div style="color:{sub_color}; font-size:0.78rem; margin-top:0.5rem; line-height:1.35;
                    overflow-wrap:break-word;">{sub_html}</div>
    </div>
    """


def _health_critical_stats(df, m) -> tuple[int, float] | None:
    """Real Critical count/percentage from the health-score distribution.

    Reuses ``get_health_distribution``'s existing health_score banding
    (Critical/Watch/Good) -- the same classification the health
    distribution chart uses -- so this never touches or reinterprets the
    composite LOW/MEDIUM/HIGH ML risk level. Only the percentage is
    recomputed here (at 2 decimals from the real count/total) rather than
    reused from ``get_health_distribution``, which rounds to 1 decimal for
    its own chart labels.
    """
    if df is None or m is None:
        return None
    distribution = get_health_distribution(df)
    critical_rows = distribution[distribution["category"] == "Critical"]
    critical_count = int(critical_rows["count"].iloc[0]) if not critical_rows.empty else 0
    total = m["total_components"]
    critical_pct = round((critical_count / total) * 100, 2) if total else 0.0
    return critical_count, critical_pct


def render_kpi_row() -> None:
    df = get_analyzed_dataset()
    m = get_kpi_metrics(df) if df is not None else None
    health_critical = _health_critical_stats(df, m)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _kpi_card(
                "TOTAL COMPONENTS",
                f"{m['total_components']:,}" if m else "—",
                f"{m['total_components_pct']:.0f}% of dataset" if m else "Analyze a dataset to populate metrics",
                _TEXT_MUTED,
                "\U0001F9E9",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _kpi_card(
                "HIGH RISK",
                f"{m['high_risk']:,}" if m else "—",
                f"{m['high_risk_pct']}% of total" if m else "Composite HIGH risk level",
                _WATCH,
                "\u26A0\uFE0F",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        # Health-score-based Critical count (Critical/Watch/Good banding),
        # deliberately distinct from the composite LOW/MEDIUM/HIGH ML risk
        # level shown in the HIGH RISK card above. See
        # ``_health_critical_stats`` for where the real count/percentage
        # come from.
        st.markdown(
            _kpi_card(
                "HEALTH CRITICAL",
                f"{health_critical[0]:,}" if health_critical else "—",
                f"{health_critical[1]:.2f}% of components"
                if health_critical
                else "Analyze a dataset to populate metrics",
                _CRITICAL,
                "\U0001F6A8",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _kpi_card(
                "SYSTEM HEALTH",
                (
                    f'{m["system_health"]}<span style="color:{_TEXT_MUTED}; font-size:1.1rem; '
                    f'font-weight:600;"> /{m["system_health_max"]}</span>'
                    if m
                    else "—"
                ),
                "Overall Reliability Score" if m else "Analyze a dataset to populate metrics",
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