"""
charts.py
---------
Plotly figures for the Overview page. Colors are pulled from
ui.theme.CHART_COLORS / COLORS so the donut, legend, and trend line all
stay visually consistent with the rest of the UI, and all figure text
is given an explicit color (never left to inherit a default that could
match the transparent/dark plot background).
"""

import plotly.graph_objects as go
import streamlit as st

from mock.mock_data import (
    HEALTH_TREND_OPTIONS,
    TOTAL_COMPONENTS,
    get_health_distribution,
    get_health_trend,
)
from ui.theme import CHART_COLORS, COLORS

PAPER_BG = "rgba(0,0,0,0)"
TEXT_COLOR = COLORS["text_primary"]
MUTED_COLOR = COLORS["text_secondary"]
GRID_COLOR = COLORS["border"]


def render_health_distribution() -> None:
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; color:{TEXT_COLOR};">'
        f'Component Health Distribution</div>',
        unsafe_allow_html=True,
    )

    df = get_health_distribution()

    chart_col, legend_col = st.columns([1.3, 1])

    with chart_col:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=df["category"],
                    values=df["count"],
                    hole=0.68,
                    marker=dict(
                        colors=[CHART_COLORS[c] for c in df["category"]],
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
                    text=f"<b style='font-size:26px;color:{TEXT_COLOR}'>{TOTAL_COMPONENTS:,}</b>"
                    f"<br><span style='font-size:12px;color:{MUTED_COLOR}'>Total</span>",
                    showarrow=False,
                    font=dict(color=TEXT_COLOR),
                )
            ],
            font=dict(color=TEXT_COLOR),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with legend_col:
        st.markdown(
            f"""
            <div style="display:flex; font-size:0.7rem; font-weight:700; color:{MUTED_COLOR};
                        text-transform:uppercase; letter-spacing:0.04em; padding:0.2rem 0.1rem;
                        border-bottom:1px solid {GRID_COLOR}; margin-bottom:0.35rem;">
                <div style="flex:1.4;">Category</div>
                <div style="flex:1; text-align:right;">Count</div>
                <div style="flex:1; text-align:right;">Percent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for _, row in df.iterrows():
            color = CHART_COLORS[row["category"]]
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; font-size:0.85rem;
                            padding:0.32rem 0.1rem; color:{TEXT_COLOR};">
                    <div style="flex:1.4; display:flex; align-items:center; gap:0.45rem;">
                        <span style="width:9px; height:9px; border-radius:50%;
                                     background:{color}; display:inline-block;"></span>
                        {row['category']}
                    </div>
                    <div style="flex:1; text-align:right; font-weight:700;">{row['count']:,}</div>
                    <div style="flex:1; text-align:right; color:{MUTED_COLOR};">{row['percentage']}%</div>
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
    header_col, dropdown_col = st.columns([2.4, 1])
    with header_col:
        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; color:{TEXT_COLOR};">'
            f'Health Score Trend</div>',
            unsafe_allow_html=True,
        )
    with dropdown_col:
        view = st.selectbox(
            "View",
            options=HEALTH_TREND_OPTIONS,
            index=0,
            key="health_trend_view",
            label_visibility="collapsed",
        )

    df = get_health_trend(view)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["hours"],
                y=df["health_score"],
                mode="lines+markers+text",
                line=dict(color=COLORS["accent"], width=3, shape="spline"),
                marker=dict(color=COLORS["accent"], size=8, line=dict(color=TEXT_COLOR, width=1)),
                text=[str(v) for v in df["health_score"]],
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
        f'Average system health score over burn-in time</div>',
        unsafe_allow_html=True,
    )