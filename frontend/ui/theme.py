"""
theme.py
--------
Single place for all colors + the global CSS injected into the app.

IMPORTANT (contrast rule):
Every text color declared below is explicitly paired with a background
color that has enough contrast against it. Never reuse a "text" color
as a "background" color (or vice versa) without checking this file
first -- that mismatch is what causes text to visually disappear.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# COLOR TOKENS
# ---------------------------------------------------------------------------

COLORS = {
    # Backgrounds (darkest -> lightest)
    "bg_app": "#0A0E17",
    "bg_sidebar": "#0A0E17",
    "bg_card": "#121826",
    "bg_card_alt": "#161D2E",
    "bg_input": "#1A2233",
    "border": "#232C40",
    "border_soft": "#1C2436",

    # Text
    "text_primary": "#F3F5F9",
    "text_secondary": "#A6B0C3",
    "text_muted": "#6C7890",
    "text_on_accent": "#FFFFFF",

    # Brand / accent
    "accent": "#6C63FF",
    "accent_soft": "#4F46E5",
    "accent_bg": "#1E1B3A",

    # Status colors -- each is a (background, text) PAIR with verified
    # contrast, used together for badges/pills. Never mix across pairs.
    "status_normal_bg": "#0F2E22",
    "status_normal_text": "#4ADE80",

    "status_watch_bg": "#3A2B06",
    "status_watch_text": "#FBBF24",

    "status_high_bg": "#3A1F08",
    "status_high_text": "#FB923C",

    "status_critical_bg": "#3D1220",
    "status_critical_text": "#FB7185",

    "status_good_bg": "#0F2E22",
    "status_good_text": "#4ADE80",

    "status_atrisk_bg": "#3A2B06",
    "status_atrisk_text": "#FBBF24",

    "status_online_bg": "#0F2E22",
    "status_online_text": "#4ADE80",
}

# Chart-specific palette (kept identical to badge accent colors so the
# donut / legend / badges all agree visually).
CHART_COLORS = {
    "Normal": "#22C55E",
    "Watch": "#FBBF24",
    "High Risk": "#FB923C",
    "Critical": "#FB7185",
}


def inject_global_css() -> None:
    c = COLORS
    # The sidebar is meant to be permanently visible for now (no
    # show/hide feature yet). Force it open with !important so it can't
    # get stuck in Streamlit's own internal "collapsed" state, and hide
    # Streamlit's native collapse handle so nobody can accidentally
    # collapse it with no way back.
    sidebar_display_css = """
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        margin-left: 0px !important;
        min-width: 244px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        transform: none !important;
        margin-left: 0px !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    """
    st.markdown(
        f"""
        <style>
        {sidebar_display_css}
        /* ---------- App shell ---------- */
        .stApp {{
            background-color: {c['bg_app']};
            color: {c['text_primary']};
        }}
        /* Keep the header bar itself (it holds the sidebar collapse arrow),
           but hide only the dev-tool clutter inside it: Deploy button,
           "..." menu, running-status spinner, and the colored decoration
           strip. Extra top padding on .block-container guarantees our
           content never renders underneath the bar, even if a future
           Streamlit version changes its exact height. */
        [data-testid="stToolbar"] {{
            visibility: hidden;
            display: none;
        }}
        [data-testid="stDecoration"] {{
            display: none;
        }}
        [data-testid="stStatusWidget"] {{
            visibility: hidden;
            display: none;
        }}
        #MainMenu {{
            visibility: hidden;
        }}
        [data-testid="stHeader"] {{
            background-color: {c['bg_app']};
        }}
        .block-container {{
            padding-top: 4.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }}
        html, body, [class*="css"] {{
            color: {c['text_primary']};
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background-color: {c['bg_sidebar']};
            border-right: 1px solid {c['border']};
        }}
        [data-testid="stSidebar"] * {{
            color: {c['text_primary']};
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background-color: transparent;
            color: {c['text_secondary']};
            border: none;
            text-align: left;
            width: 100%;
            font-weight: 500;
            padding: 0.55rem 0.9rem;
            border-radius: 8px;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: {c['bg_card_alt']};
            color: {c['text_primary']};
            border: none;
        }}
        [data-testid="stSidebar"] .stButton > button:focus:not(:active) {{
            color: {c['text_primary']};
        }}

        /* Sidebar file uploader text visibility */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background-color: {c['bg_input']};
            border: 1px dashed {c['border']};
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
            color: {c['text_secondary']} !important;
        }}

        /* ---------- Generic text elements ---------- */
        p, span, label, li, div {{
            color: inherit;
        }}
        .muted {{
            color: {c['text_secondary']};
        }}

        /* ---------- Buttons (main area) ---------- */
        .stButton > button, .stDownloadButton > button {{
            background-color: {c['accent']};
            color: {c['text_on_accent']} !important;
            border: 1px solid {c['accent']};
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {c['accent_soft']};
            border-color: {c['accent_soft']};
            color: {c['text_on_accent']} !important;
        }}
        .stButton > button p {{
            color: {c['text_on_accent']} !important;
        }}

        /* Secondary / ghost button variant */
        .btn-secondary button {{
            background-color: {c['bg_card_alt']} !important;
            color: {c['text_primary']} !important;
            border: 1px solid {c['border']} !important;
        }}
        .btn-secondary button:hover {{
            background-color: {c['bg_input']} !important;
        }}

        /* ---------- Select boxes / dropdowns ---------- */
        [data-baseweb="select"] > div {{
            background-color: {c['bg_input']} !important;
            border-color: {c['border']} !important;
            color: {c['text_primary']} !important;
        }}
        [data-baseweb="select"] * {{
            color: {c['text_primary']} !important;
        }}
        [data-baseweb="popover"] * {{
            color: {c['text_primary']} !important;
        }}
        ul[role="listbox"] {{
            background-color: {c['bg_card']} !important;
        }}

        /* ---------- Dataframe / table default widget (fallback) ---------- */
        [data-testid="stDataFrame"] {{
            background-color: {c['bg_card']};
        }}

        /* ---------- Custom component classes ---------- */
        .app-title {{
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: {c['text_primary']};
            margin-bottom: 0;
        }}
        .app-title .accent {{
            color: {c['accent']};
        }}
        .app-tagline {{
            color: {c['text_muted']};
            font-size: 0.8rem;
            margin-top: -6px;
            margin-bottom: 1.1rem;
        }}

        .page-title {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {c['text_primary']};
            margin-bottom: 0.1rem;
        }}
        .page-subtitle {{
            color: {c['text_secondary']};
            font-size: 0.95rem;
            margin-bottom: 0;
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {c['text_primary']};
            margin-bottom: 0.1rem;
        }}
        .section-caption {{
            color: {c['text_muted']};
            font-size: 0.78rem;
        }}

        .card {{
            background-color: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
        }}

        .kpi-card {{
            background-color: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 14px;
            padding: 1.0rem 1.15rem;
            height: 100%;
        }}
        .kpi-label {{
            color: {c['text_muted']};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .kpi-value {{
            color: {c['text_primary']};
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.15;
            margin-top: 0.15rem;
        }}
        .kpi-value .kpi-value-max {{
            color: {c['text_muted']};
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .kpi-sub {{
            color: {c['text_muted']};
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }}
        .kpi-sub.positive {{ color: {c['status_normal_text']}; }}
        .kpi-sub.warning {{ color: {c['status_watch_text']}; }}
        .kpi-sub.danger {{ color: {c['status_critical_text']}; }}

        /* Status pill used for SYSTEM STATUS: ONLINE */
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background-color: {c['status_online_bg']};
            color: {c['status_online_text']};
            border: 1px solid {c['status_online_bg']};
        }}
        .pill-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: {c['status_online_text']};
            display: inline-block;
        }}

        /* Risk / status badges (table cells) */
        .badge {{
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }}
        .badge-high {{
            background-color: {c['status_high_bg']};
            color: {c['status_high_text']};
        }}
        .badge-watch {{
            background-color: {c['status_watch_bg']};
            color: {c['status_watch_text']};
        }}
        .badge-critical {{
            background-color: {c['status_critical_bg']};
            color: {c['status_critical_text']};
        }}
        .badge-good {{
            background-color: {c['status_good_bg']};
            color: {c['status_good_text']};
        }}
        .badge-atrisk {{
            background-color: {c['status_atrisk_bg']};
            color: {c['status_atrisk_text']};
        }}

        /* Custom HTML table used for high-risk / lot tables */
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
        }}
        table.data-table thead th {{
            text-align: left;
            color: {c['text_muted']};
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.5rem 0.6rem;
            border-bottom: 1px solid {c['border']};
        }}
        table.data-table tbody td {{
            color: {c['text_primary']};
            padding: 0.6rem 0.6rem;
            border-bottom: 1px solid {c['border_soft']};
        }}
        table.data-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        table.data-table tbody tr:hover {{
            background-color: {c['bg_card_alt']};
        }}
        .cell-muted {{
            color: {c['text_secondary']};
        }}
        .cell-strong {{
            color: {c['text_primary']};
            font-weight: 700;
        }}

        /* Dataset info block (sidebar) */
        .dataset-box {{
            background-color: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }}
        .dataset-label {{
            color: {c['text_muted']};
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .dataset-filename {{
            color: {c['status_normal_text']};
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.15rem;
            margin-bottom: 0.6rem;
            word-break: break-all;
        }}
        .dataset-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            padding: 0.18rem 0;
            color: {c['text_secondary']};
        }}
        .dataset-row span:last-child {{
            color: {c['text_primary']};
            font-weight: 600;
        }}

        /* CTA banner */
        .cta-banner {{
            background: linear-gradient(90deg, {c['accent_bg']} 0%, {c['bg_card']} 100%);
            border: 1px solid {c['border']};
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
        }}
        .cta-title {{
            color: {c['text_primary']};
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }}
        .cta-desc {{
            color: {c['text_secondary']};
            font-size: 0.88rem;
        }}

        /* Coming soon page */
        .coming-soon-wrap {{
            text-align: center;
            padding: 5rem 1rem;
        }}
        .coming-soon-title {{
            color: {c['text_primary']};
            font-size: 1.5rem;
            font-weight: 800;
        }}
        .coming-soon-sub {{
            color: {c['text_muted']};
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }}

        hr {{
            border-color: {c['border']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )