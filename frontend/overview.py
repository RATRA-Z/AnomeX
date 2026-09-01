"""
overview.py
-----------
The Overview dashboard page. Pure composition -- all data comes from
mock.mock_data (via the ui/* components), no numbers are typed here.
"""

import streamlit as st

from ui.cards import render_cta_banner, render_header, render_kpi_row
from ui.charts import render_health_distribution, render_health_trend
from ui.tables import render_high_risk_table, render_lot_summary_table

_CARD_OPEN = (
    '<div style="background-color:#121826; border:1px solid #232C40; '
    'border-radius:14px; padding:1.1rem 1.25rem;">'
)
_CARD_CLOSE = "</div>"


def render() -> None:
    render_header()

    st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)
    render_kpi_row()

    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.markdown(_CARD_OPEN, unsafe_allow_html=True)
        render_health_distribution()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown(_CARD_OPEN, unsafe_allow_html=True)
        render_health_trend()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(_CARD_OPEN, unsafe_allow_html=True)
        render_high_risk_table()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown(_CARD_OPEN, unsafe_allow_html=True)
        render_lot_summary_table()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
    render_cta_banner()