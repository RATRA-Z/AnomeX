"""
sidebar.py
----------
Renders the permanent left sidebar: brand header, navigation, and the
"current dataset" panel with a CSV uploader.

Navigation state lives in st.session_state["active_page"] so app.py can
decide which page module to render.

NOTE: the dataset panel below uses INLINE styles (not the .dataset-box /
.dataset-row classes from ui/theme.py) so it renders correctly even if
the global stylesheet from inject_global_css() hasn't loaded for any
reason -- this was causing unstyled text to run together (e.g.
"Rows10,000") in some environments.
"""

import streamlit as st

from mock.mock_data import DATASET_INFO

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
_ACCENT = "#6C63FF"


def render_sidebar() -> None:
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
    info = DATASET_INFO

    rows_html = "".join(
        [
            _dataset_row("Rows", f"{info['rows']:,}"),
            _dataset_row("Components", f"{info['components']:,}"),
            _dataset_row("Lots", f"{info['lots']}"),
            _dataset_row("Primary Parameter", info["primary_parameter"]),
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
            <div style="color:{_NORMAL}; font-size:0.85rem; font-weight:600;
                        margin:0.2rem 0 0.6rem 0; word-break:break-all;">
                \U0001F4C4 {info['file_name']}
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)

    if st.button("\u2B06\uFE0F  Replace Dataset", key="replace_dataset_btn", use_container_width=True):
        st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)

    if st.session_state.get("show_uploader", False):
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="sidebar_csv_uploader",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.session_state["uploaded_file_name"] = uploaded.name
            st.success(f"Loaded: {uploaded.name} (demo mode - not analyzed yet)")