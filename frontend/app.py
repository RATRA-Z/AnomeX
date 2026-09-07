"""
app.py
------
AnomeX - AI-Driven Anomaly Detection in Component Burn-In and Screening

Application entry point. It initializes the frontend session lifecycle,
then wires the sidebar and active page together. Dataset validation and ML
orchestration remain inside data.session rather than this module.
"""

import sys
from pathlib import Path

import streamlit as st

# Running ``streamlit run frontend/app.py`` adds frontend/ to sys.path but not
# the repository root. The root is required by data.session's existing ml/*
# imports, so retain this narrow bootstrap path adjustment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.session import initialize_session_state
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


def _initialize_application_session() -> None:
    """Prepare empty session state for a fresh session.

    No dataset is read and no analysis is run here. A brand-new session
    starts with every session.py key at its empty default (raw_dataset,
    analyzed_dataset, etc. all None/False) so the dashboard shows a
    no-dataset/ready-to-upload state until the user explicitly uploads a
    CSV and clicks Analyze Dataset. ``initialize_session_state()`` only
    fills in missing keys via ``setdefault`` (see data/session.py), so
    calling it on every rerun never resets a dataset that has already
    been uploaded and/or analyzed -- it's a no-op once those keys exist.
    ``initialize_default_dataset()`` and ``reset_to_default_dataset()``
    remain available in session.py for explicit development/reset use;
    they are simply no longer invoked automatically from here.
    """
    initialize_session_state()


_initialize_application_session()
render_sidebar()

active_page = st.session_state.get("active_page", "Overview")

if active_page == "Overview":
    overview.render()
else:
    coming_soon.render(active_page)