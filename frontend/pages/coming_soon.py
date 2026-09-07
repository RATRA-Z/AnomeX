"""
coming_soon.py
--------------
Simple placeholder page shown for every sidebar nav item other than
Overview (Components, System Intelligence, What-If Simulator,
AnomeX AI, Settings). No backend logic here on purpose.
"""

import streamlit as st

_ICONS = {
    "Components": "\U0001F5C2\uFE0F",
    "System Intelligence": "\u2699\uFE0F",
    "What-If Simulator": "\U0001F52C",
    "AnomeX AI": "\U0001F4A1",
    "Settings": "\u2699\uFE0F",
}


def render(page_name: str) -> None:
    icon = _ICONS.get(page_name, "\U0001F6A7")
    st.markdown(
        f"""
        <div style="text-align:center; padding:5rem 1rem;">
            <div style="font-size:2.4rem;">{icon}</div>
            <div style="color:#F3F5F9; font-size:1.5rem; font-weight:800;">
                {page_name} — Coming Soon
            </div>
            <div style="color:#6C7890; font-size:0.95rem; margin-top:0.4rem;">
                This section will be wired up once the backend / ML pipeline is ready.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("\u2190 Back to Overview", use_container_width=True):
            st.session_state["active_page"] = "Overview"
            st.rerun()