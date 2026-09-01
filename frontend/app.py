"""
app.py
------
AnomeX - AI-Driven Anomaly Detection in Component Burn-In and Screening

Frontend-only entry point. Wires up the sidebar and routes to the
active page. No AI/ML, backend, or database logic lives here -- all
data is placeholder data from mock/mock_data.py, structured so it can
be swapped for real Pandas DataFrames / backend results later.
"""

import streamlit as st

from pages import coming_soon, overview
from ui.sidebar import render_sidebar
from ui.theme import inject_global_css

st.set_page_config(
    page_title="AnomeX | Dashboard Overview",
    page_icon="\U0001F9E9",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
render_sidebar()

active_page = st.session_state.get("active_page", "Overview")

if active_page == "Overview":
    overview.render()
else:
    coming_soon.render(active_page)