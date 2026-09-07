"""
charts.py
---------
Plotly figures for the Overview page. Colors are pulled from
ui.theme.CHART_COLORS / COLORS so the donut, legend, and trend line all
stay visually consistent with the rest of the UI, and all figure text
is given an explicit color (never left to inherit a default that could
match the transparent/dark plot background).

Category color lookups always go through `_category_color()` — never
direct dict indexing — so a category the current theme palette doesn't
happen to define (e.g. "Good") can never raise a KeyError.
"""

import plotly.graph_objects as go
import streamlit as st

from data.dashboard_data import (
    get_health_distribution,
    get_health_trend,
)
from data.session import get_analyzed_dataset
from ui.theme import CHART_COLORS, COLORS

PAPER_BG = "rgba(0,0,0,0)"
TEXT_COLOR = COLORS["text_primary"]
MUTED_COLOR = COLORS["text_secondary"]
GRID_COLOR = COLORS["border"]

# Absolute last-resort color if a category is missing from CHART_COLORS
# AND the theme has no "Normal" fallback entry either. Reuses an
# already-existing theme color (never invents a new one).
_ULTIMATE_FALLBACK_COLOR = MUTED_COLOR


def _category_color(category: str) -> str:
    """
    Safe color lookup for any health-distribution category
    (Critical / Watch / Good, or whatever the dataset actually
    produces). Never assumes a specific key exists in CHART_COLORS:

        1. Use the category's own entry if the palette defines one.
        2. Otherwise fall back to the palette's "Normal" entry, if any.
        3. Otherwise fall back to a neutral existing theme color.

    This replaces the previous direct `CHART_COLORS[category]`
    indexing in the legend (the source of `KeyError: 'Good'`) with the
    same safe pattern already used for the donut slices.
    """
    if category in CHART_COLORS:
        return CHART_COLORS[category]
    if "Normal" in CHART_COLORS:
        return CHART_COLORS["Normal"]
    return _ULTIMATE_FALLBACK_COLOR


def render_health_distribution() -> None:
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; color:{TEXT_COLOR};">'
        f'Component Health Distribution</div>',
        unsafe_allow_html=True,
    )

    data = get_analyzed_dataset()
    if data is None or data.empty:
        st.info("Analyze the current dataset to display health distribution.")
        return

    try:
        df = get_health_distribution(data)
    except Exception:
        st.info("Health distribution is not available for this dataset yet.")
        return

    if df is None or df.empty or "category" not in df.columns:
        st.info("No health distribution data available.")
        return

    # Only keep categories that actually have components, so the donut
    # never renders a zero-value slice — a category being absent (0
    # components) is expected and must not crash or clutter the chart.
    df = df[df["count"] > 0].reset_index(drop=True)
    total_components = len(data)

    if df.empty:
        st.info("No components fall into a health category yet.")
        return

    chart_col, legend_col = st.columns([1.3, 1])

    with chart_col:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=df["category"],
                    values=df["count"],
                    hole=0.68,
                    marker=dict(
                        colors=[_category_color(c) for c in df["category"]],
                        line=dict(color=COLORS["bg_card"], width=3),
                    ),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
                    sort=False,
                )
            ]
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PAPER_BG,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            annotations=[
                dict(
                    text=f"<b style='font-size:26px;color:{TEXT_COLOR}'>{total_components:,}</b>"
                    f"<br><span style='font-size:12px;color:{MUTED_COLOR}'>Total</span>",
                    showarrow=False,
                    font=dict(color=TEXT_COLOR),
                )
            ],
            font=dict(color=TEXT_COLOR),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with legend_col:
        legend_rows_html = "".join(
            f"""
            <div style="display:flex; align-items:center; font-size:0.85rem;
                        padding:0.32rem 0.1rem; color:{TEXT_COLOR};">
                <div style="flex:1.4; display:flex; align-items:center; gap:0.45rem;">
                    <span style="width:9px; height:9px; border-radius:50%;
                                 background:{_category_color(row['category'])}; display:inline-block;"></span>
                    {row['category']}
                </div>
                <div style="flex:1; text-align:right; font-weight:700;">{row['count']:,}</div>
                <div style="flex:1; text-align:right; color:{MUTED_COLOR};">{row['percentage']}%</div>
            </div>
            """
            for _, row in df.iterrows()
        )
        # Legend is wrapped to the same height as the donut (260px) and
        # vertically centered, so the two columns line up cleanly
        # regardless of how many categories are present (1, 2, or 3).
        st.markdown(
            f"""
            <div style="min-height:260px; display:flex; flex-direction:column; justify-content:center;">
                <div style="display:flex; font-size:0.7rem; font-weight:700; color:{MUTED_COLOR};
                            text-transform:uppercase; letter-spacing:0.04em; padding:0.2rem 0.1rem;
                            border-bottom:1px solid {GRID_COLOR}; margin-bottom:0.35rem;">
                    <div style="flex:1.4;">Category</div>
                    <div style="flex:1; text-align:right;">Count</div>
                    <div style="flex:1; text-align:right;">Percent</div>
                </div>
                {legend_rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="color:{MUTED_COLOR}; font-size:0.78rem; margin-top:0.4rem;">'
        f'Distribution of components based on health score</div>',
        unsafe_allow_html=True,
    )


def render_health_trend() -> None:
    # NOTE: previously wrapped in its own `st.container()`, but that
    # container held only this title markdown and served no layout
    # purpose -- removed as a harmless simplification. It was not,
    # by itself, the source of the empty rounded box reported above
    # both chart sections: render_health_distribution()'s title above
    # has no wrapping container at all, yet reportedly shows the same
    # box, so that box is coming from outside this file (most likely
    # a leftover wrapper in overview.py around each section, or a
    # global CSS rule in theme.py) -- see chat reply for details.
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; color:{TEXT_COLOR};">'
        f'Health Score Trend (Derived from Measurements)</div>',
        unsafe_allow_html=True,
    )

    data = get_analyzed_dataset()
    if data is None or data.empty:
        st.info("Analyze the current dataset to display the health trend.")
        return

    try:
        df = get_health_trend(data)
    except Exception:
        st.info("Health trend is not available for this dataset yet.")
        return

    if df is None or df.empty or "hours" not in df.columns:
        st.info("No health trend data available.")
        return

    # The trend column is whatever data.dashboard_data.get_health_trend()
    # actually names its derived value column (e.g. "health_score").
    # Looked up by name rather than assumed, so this keeps working even
    # if that column is ever renamed on the data side (still owned by
    # dashboard_data.py, not decided here).
    value_column = "health_score" if "health_score" in df.columns else next(
        (c for c in df.columns if c != "hours"), None
    )
    if value_column is None or df[value_column].isna().all():
        st.info("No health trend data available.")
        return

    y_values = df[value_column]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["hours"],
                y=y_values,
                mode="lines+markers+text",
                line=dict(color=COLORS["accent"], width=3, shape="spline"),
                marker=dict(color=COLORS["accent"], size=8, line=dict(color=TEXT_COLOR, width=1)),
                text=[str(v) for v in y_values],
                textposition="top center",
                textfont=dict(color=TEXT_COLOR, size=12),
                fill="tozeroy",
                fillcolor="rgba(108, 99, 255, 0.12)",
                hovertemplate="%{x}h: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        margin=dict(l=10, r=20, t=30, b=10),
        height=260,
        font=dict(color=MUTED_COLOR),
        xaxis=dict(
            title=dict(text="Burn-In Time (Hours)", font=dict(color=MUTED_COLOR, size=11)),
            tickmode="array",
            tickvals=df["hours"].tolist(),
            ticktext=[f"{h}h" for h in df["hours"]],
            tickfont=dict(color=MUTED_COLOR),
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Health Score", font=dict(color=MUTED_COLOR, size=11)),
            range=[0, 105],
            tickfont=dict(color=MUTED_COLOR),
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="color:{MUTED_COLOR}; font-size:0.78rem; margin-top:0.4rem;">'
        f'Derived measurement-based trend relative to the 0h baseline; not a model-generated checkpoint health prediction.</div>',
        unsafe_allow_html=True,
    )