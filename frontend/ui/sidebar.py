"""
sidebar.py
----------
Renders the permanent left sidebar: brand header, navigation, and the
current-dataset control panel.

Navigation state lives in st.session_state["active_page"] so app.py can
decide which page module to render.

NOTE: the dataset panel below uses INLINE styles (not the .dataset-box /
.dataset-row classes from ui/theme.py) so it renders correctly even if
the global stylesheet from inject_global_css() hasn't loaded for any
reason -- this was causing unstyled text to run together (e.g.
"Rows10,000") in some environments.
"""

import pandas as pd
import streamlit as st

from data.session import (
    ANALYSIS_ERROR_KEY,
    ANALYSIS_READY_KEY,
    DATASET_FILENAME_KEY,
    DATASET_SOURCE_KEY,
    accept_uploaded_dataset,
    get_raw_dataset,
    get_validation_result,
    initialize_session_state,
    reset_to_default_dataset,
    run_analysis,
)
from data.validation import DatasetValidationResult

NAV_ITEMS = [
    ("Overview", "\U0001F3E0"),
    ("Components", "\U0001F5C2\uFE0F"),
    ("System Intelligence", "\u2699\uFE0F"),
    ("What-If Simulator", "\U0001F52C"),
    ("AnomeX AI", "\U0001F4A1"),
    ("Settings", "\u2699\uFE0F"),
]

_TEXT_PRIMARY = "#F3F5F9"
_TEXT_SECONDARY = "#A6B0C3"
_TEXT_MUTED = "#6C7890"
_BG_CARD = "#121826"
_BORDER = "#232C40"
_NORMAL = "#4ADE80"
_WATCH = "#FBBF24"
_ACCENT = "#6C63FF"


def render_sidebar() -> None:
    initialize_session_state()
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"

    with st.sidebar:
        st.markdown(
            f"""
            <div style="font-size:1.4rem; font-weight:800; letter-spacing:-0.02em;
                        color:{_TEXT_PRIMARY};">
                Anome<span style="color:{_ACCENT};">X</span>
            </div>
            <div style="color:{_TEXT_MUTED}; font-size:0.78rem; margin:0 0 1.1rem 0;">
                Predict. Detect. Prevent.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for label, icon in NAV_ITEMS:
            is_active = st.session_state["active_page"] == label
            btn_label = f"{icon}  {label}"
            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        background-color:{_ACCENT};
                        color:#FFFFFF;
                        border-radius:8px;
                        padding:0.55rem 0.9rem;
                        font-weight:700;
                        margin-bottom:0.15rem;
                        font-size:0.9rem;">
                        {btn_label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if st.button(btn_label, key=f"nav_{label}", use_container_width=True):
                    st.session_state["active_page"] = label
                    st.rerun()

        st.markdown(
            f"<hr style='margin:1.1rem 0; border:none; border-top:1px solid {_BORDER};'>",
            unsafe_allow_html=True,
        )

        _render_dataset_panel()


def _dataset_row(label: str, value: str) -> str:
    """One label/value row built with inline flex styles (no external class)."""
    return f"""
    <div style="display:flex; justify-content:space-between; align-items:baseline;
                font-size:0.82rem; padding:0.2rem 0; color:{_TEXT_SECONDARY};">
        <span>{label}</span>
        <span style="color:{_TEXT_PRIMARY}; font-weight:600; margin-left:0.5rem;">{value}</span>
    </div>
    """


def _render_dataset_panel() -> None:
    raw_dataset = get_raw_dataset()
    source = st.session_state.get(DATASET_SOURCE_KEY)
    filename = st.session_state.get(DATASET_FILENAME_KEY) or "No dataset selected"
    is_uploaded = source == "uploaded"
    source_label = "Uploaded Dataset" if is_uploaded else "Default Dataset"
    source_color = _WATCH if is_uploaded else _NORMAL
    row_count = len(raw_dataset) if isinstance(raw_dataset, pd.DataFrame) else 0
    component_count = (
        raw_dataset["component_id"].nunique()
        if isinstance(raw_dataset, pd.DataFrame) and "component_id" in raw_dataset.columns
        else 0
    )
    lot_count = (
        raw_dataset["lot_id"].nunique()
        if isinstance(raw_dataset, pd.DataFrame) and "lot_id" in raw_dataset.columns
        else 0
    )
    analysis_ready = bool(st.session_state.get(ANALYSIS_READY_KEY, False))
    analysis_status = "Analysis Ready" if analysis_ready else "Ready for Analysis"
    validation_result = get_validation_result()

    rows_html = "".join(
        [
            _dataset_row("Source", source_label),
            _dataset_row("Rows", f"{row_count:,}"),
            _dataset_row("Components", f"{component_count:,}"),
            _dataset_row("Lots", f"{lot_count:,}"),
            _dataset_row("Status", analysis_status),
        ]
    )

    st.markdown(
        f"""
        <div style="background-color:{_BG_CARD}; border:1px solid {_BORDER};
                    border-radius:12px; padding:0.9rem 1rem;">
            <div style="color:{_TEXT_MUTED}; font-size:0.68rem; font-weight:700;
                        letter-spacing:0.06em; text-transform:uppercase;">
                Current Dataset
            </div>
            <div style="color:{source_color}; font-size:0.85rem; font-weight:600;
                        margin:0.2rem 0 0.6rem 0; word-break:break-all;">
                \U0001F4C4 {filename}
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)

    _render_sidebar_notice()

    if st.button("\u2B06\uFE0F  Replace Dataset", key="replace_dataset_btn", use_container_width=True):
        st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)

    if st.session_state.get("show_uploader", False):
        st.caption("Upload a CSV, then explicitly load it for validation.")
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="sidebar_csv_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.caption(f"Selected: {uploaded.name}")
            if st.button("Load Uploaded Dataset", key="load_uploaded_dataset", use_container_width=True):
                result = accept_uploaded_dataset(uploaded, filename=uploaded.name)
                if result.is_valid:
                    st.session_state["show_uploader"] = False
                    _set_sidebar_notice(
                        f"Dataset loaded: {uploaded.name} — {result.row_count:,} rows. Ready for analysis."
                    )
                    st.rerun()
                _render_validation_feedback(result)

    if st.button("\u25B6\uFE0F  Analyze Dataset", key="sidebar_analyze_dataset", use_container_width=True):
        with st.spinner("Running AnomeX analysis..."):
            analyzed = run_analysis()
        if analyzed is not None:
            _set_sidebar_notice(f"Analysis complete for {len(analyzed):,} components.")
            st.rerun()
        else:
            error = st.session_state.get(ANALYSIS_ERROR_KEY, "Analysis could not be completed.")
            st.error(error)

    if st.button("↺  Reset Dataset", key="reset_dataset_btn", use_container_width=True):
        with st.spinner("Restoring and analyzing the default dataset..."):
            result = reset_to_default_dataset()
        if result.is_valid and st.session_state.get(ANALYSIS_READY_KEY, False):
            _set_sidebar_notice("Default dataset restored and analysis completed.")
            st.rerun()
        else:
            error = st.session_state.get(ANALYSIS_ERROR_KEY, "Default dataset could not be restored.")
            st.error(error)

    if validation_result is not None:
        _render_validation_feedback(validation_result)
    if st.session_state.get(ANALYSIS_ERROR_KEY):
        st.error(st.session_state[ANALYSIS_ERROR_KEY])


def _render_validation_feedback(result: DatasetValidationResult) -> None:
    """Render controller-provided validation feedback without revalidating."""
    if result.is_valid:
        for warning in result.warnings:
            st.warning(warning)
        return

    for error in result.errors:
        st.error(error)
    if result.missing_columns:
        st.caption("Missing columns: " + ", ".join(result.missing_columns))
    for column, reasons in result.invalid_columns.items():
        st.caption(f"{column}: " + "; ".join(reasons))


def _set_sidebar_notice(message: str) -> None:
    st.session_state["sidebar_dataset_notice"] = message


def _render_sidebar_notice() -> None:
    message = st.session_state.pop("sidebar_dataset_notice", None)
    if message:
        st.success(message)