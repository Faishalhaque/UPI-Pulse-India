"""
app.py
------
UPI Pulse India — Main Streamlit Application
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
Author: Mo Faishal Haque
"""

import sys
import os

# Make src importable whether running from project root or elsewhere
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import (
    load_data,
    filter_data,
    get_annual_summary,
    MONTH_ORDER,
)
from src.analytics import (
    total_transaction_volume,
    total_transaction_value,
    average_transaction_value,
    period_over_period_growth,
    latest_period,
    trend_summary,
    detect_anomalies,
    simple_forecast,
    yoy_growth_rates,
)
from src.insights import generate_local_insights, enhance_with_llm

load_dotenv()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UPI Pulse India",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design Tokens — injected once
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #8b949e !important; font-size:12px; }

/* ── KPI Cards ── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 18px 16px;
    text-align: center;
    min-height: 110px;
    display: flex; flex-direction: column; justify-content: center;
}
.kpi-label { font-size: 11px; color: #8b949e; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #58a6ff;
             line-height: 1.2; }
.kpi-sub   { font-size: 12px; color: #3fb950; margin-top: 4px; }
.kpi-sub.down { color: #f85149; }
.kpi-sub.neutral { color: #8b949e; }

/* ── Section headers ── */
.section-header {
    font-size: 20px; font-weight: 600; color: #e6edf3;
    border-bottom: 2px solid #21262d; padding-bottom: 8px; margin-bottom: 16px;
}

/* ── Data table ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Demo banner ── */
.demo-banner {
    background: #2d1b00; border: 1px solid #d97706; border-radius: 8px;
    padding: 10px 16px; color: #fbbf24; font-size: 13px; margin-bottom: 16px;
}

/* ── Info box ── */
.info-box {
    background: #0d2137; border: 1px solid #1e4a72; border-radius: 8px;
    padding: 14px 16px; font-size: 14px; color: #79c0ff;
}

/* ── Insight card ── */
.insight-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 14px; font-size: 14px; line-height: 1.7;
}
.insight-title { font-size: 13px; font-weight: 600; color: #58a6ff;
                 text-transform: uppercase; letter-spacing: 0.06em;
                 margin-bottom: 10px; }

/* ── About page ── */
.about-section { padding: 8px 0 20px; }
.about-section h3 { color: #58a6ff; font-size: 16px; margin-bottom: 6px; }
.about-section p, .about-section li { color: #c9d1d9; font-size: 14px; line-height: 1.7; }

/* Streamlit overrides */
.stButton button {
    background: #1f6feb; color: #fff; border: none; border-radius: 6px;
    font-size: 13px; padding: 6px 16px;
}
.stButton button:hover { background: #388bfd; }
div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 14px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="-apple-system, Segoe UI, sans-serif", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#8b949e"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#8b949e"),
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
)

COLOR_VOL = "#58a6ff"
COLOR_VAL = "#3fb950"
COLOR_AVG = "#d2a8ff"
COLOR_GROWTH = "#ffa657"
COLOR_ANOMALY = "#f85149"


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Session-state: load data once, cache for session
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load(file_key):
    """Cache data loading; file_key changes when user uploads a new file."""
    return load_data()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame, meta: dict):
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 16px 0 8px;'>
              <div style='font-size:26px;'>💳</div>
              <div style='font-size:18px; font-weight:700; color:#58a6ff;'>UPI Pulse India</div>
              <div style='font-size:11px; color:#8b949e; margin-top:2px;'>
                Digital Payments Analytics
              </div>
            </div>
            <hr style='border-color:#21262d; margin:10px 0 16px;'/>
            """,
            unsafe_allow_html=True,
        )

        # Page navigation
        page = st.radio(
            "Navigation",
            [
                "📊 Overview",
                "📈 Transaction Trends",
                "💡 Payment Insights",
                "🔍 Data Explorer",
                "🤖 AI Insights",
                "ℹ️ About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#21262d; margin:16px 0;'/>", unsafe_allow_html=True)

        # --- Filters ---
        if not df.empty:
            st.markdown(
                "<div style='font-size:12px;color:#8b949e;text-transform:uppercase;"
                "letter-spacing:0.08em;margin-bottom:8px;'>Filters</div>",
                unsafe_allow_html=True,
            )
            available_years = sorted(df["year"].unique().tolist())
            selected_years = st.multiselect(
                "Year(s)",
                available_years,
                default=available_years,
                key="year_filter",
            )
            if not selected_years:
                selected_years = available_years

            periods = df["period"].tolist()
            col1, col2 = st.columns(2)
            with col1:
                start_idx = st.selectbox("From", periods, index=0, key="start_period")
            with col2:
                end_idx = st.selectbox("To", periods, index=len(periods) - 1, key="end_period")

            st.markdown(
                "<hr style='border-color:#21262d; margin:16px 0;'/>", unsafe_allow_html=True
            )

        # File upload
        st.markdown(
            "<div style='font-size:12px;color:#8b949e;text-transform:uppercase;"
            "letter-spacing:0.08em;margin-bottom:8px;'>Dataset</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload official CSV",
            type=["csv"],
            help="Upload NPCI monthly statistics CSV. "
            "Columns: month, year, transaction_volume, transaction_value",
        )

        st.markdown(
            f"<div style='font-size:11px;color:#8b949e;margin-top:8px;'>"
            f"Source: {meta.get('source','N/A')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    return page, selected_years if not df.empty else [], start_idx if not df.empty else None, end_idx if not df.empty else None, uploaded


# ---------------------------------------------------------------------------
# KPI Cards row
# ---------------------------------------------------------------------------
def render_kpi_row(df: pd.DataFrame):
    tot_vol = total_transaction_volume(df)
    tot_val = total_transaction_value(df)
    avg_val = average_transaction_value(df)
    mom = period_over_period_growth(df, "transaction_volume", 1)
    lp = latest_period(df)

    def fmt_growth(v):
        if np.isnan(v):
            return ("—", "neutral")
        sign = "↑" if v >= 0 else "↓"
        cls = "down" if v < 0 else ""
        return (f"{sign} {abs(v):.1f}% MoM", cls)

    growth_str, growth_cls = fmt_growth(mom)

    cards = [
        ("Total Transactions", f"{tot_vol:,.1f} Cr", "Transaction count in crore", ""),
        ("Total Value", f"₹{tot_val:,.2f}", "In ₹ lakh crore", ""),
        ("Avg Transaction Value", f"₹{avg_val:,.0f}" if not np.isnan(avg_val) else "—", "Per transaction", ""),
        ("MoM Volume Growth", growth_str, "Last vs previous month", growth_cls),
        ("Latest Period", lp, "Most recent data point", "neutral"),
    ]

    cols = st.columns(len(cards))
    for col, (label, value, sub, cls) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-sub {cls}">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_volume_trend(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="period",
        y="transaction_volume",
        title="Monthly UPI Transaction Volume (Crore)",
        labels={"transaction_volume": "Volume (Cr)", "period": "Period"},
        color_discrete_sequence=[COLOR_VOL],
    )
    fig.update_traces(line_width=2, mode="lines+markers", marker_size=4)
    apply_theme(fig)
    fig.update_layout(title_font_size=15)
    return fig


def chart_value_trend(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="period",
        y="transaction_value",
        title="Monthly UPI Transaction Value (₹ Lakh Crore)",
        labels={"transaction_value": "Value (₹ Lakh Cr)", "period": "Period"},
        color_discrete_sequence=[COLOR_VAL],
    )
    fig.update_traces(line_width=2, mode="lines+markers", marker_size=4)
    apply_theme(fig)
    fig.update_layout(title_font_size=15)
    return fig


def chart_vol_vs_val(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["transaction_volume"],
            name="Volume (Cr)",
            line=dict(color=COLOR_VOL, width=2),
            mode="lines+markers",
            marker_size=3,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["transaction_value"],
            name="Value (₹ Lakh Cr)",
            line=dict(color=COLOR_VAL, width=2, dash="dot"),
            mode="lines+markers",
            marker_size=3,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Volume vs Value Comparison",
        title_font_size=15,
        **PLOTLY_LAYOUT,
    )
    fig.update_yaxes(title_text="Volume (Crore)", secondary_y=False,
                     gridcolor="#21262d", tickcolor="#8b949e")
    fig.update_yaxes(title_text="Value (₹ Lakh Crore)", secondary_y=True,
                     gridcolor="#21262d", tickcolor="#8b949e")
    return fig


def chart_yoy_growth(df: pd.DataFrame) -> go.Figure:
    df2 = yoy_growth_rates(df).dropna(subset=["volume_yoy_growth"])
    if df2.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for YoY calculation (need ≥ 13 months)",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#8b949e", size=14))
        apply_theme(fig)
        fig.update_layout(title="Year-over-Year Growth (%)")
        return fig

    colors = [COLOR_VAL if v >= 0 else COLOR_ANOMALY for v in df2["volume_yoy_growth"]]
    fig = go.Figure(go.Bar(
        x=df2["period"],
        y=df2["volume_yoy_growth"],
        marker_color=colors,
        name="YoY Volume Growth %",
        text=df2["volume_yoy_growth"].round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(size=9, color="#8b949e"),
    ))
    fig.update_layout(title="Year-over-Year Transaction Volume Growth (%)",
                      title_font_size=15, **PLOTLY_LAYOUT)
    fig.add_hline(y=0, line_color="#8b949e", line_width=1, line_dash="dot")
    return fig


def chart_avg_txn_value(df: pd.DataFrame) -> go.Figure:
    df2 = df.dropna(subset=["avg_transaction_value"])
    fig = px.area(
        df2,
        x="period",
        y="avg_transaction_value",
        title="Average Transaction Value per UPI Transfer (₹)",
        labels={"avg_transaction_value": "Avg Value (₹)", "period": "Period"},
        color_discrete_sequence=[COLOR_AVG],
    )
    fig.update_traces(line_width=2, fillcolor="rgba(210,168,255,0.12)")
    apply_theme(fig)
    fig.update_layout(title_font_size=15)
    return fig


def chart_annual_summary(annual_df: pd.DataFrame) -> go.Figure:
    if annual_df.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Annual Transaction Volume (Crore)", "Annual Transaction Value (₹ Lakh Crore)"),
    )
    fig.add_trace(
        go.Bar(x=annual_df["year"].astype(str), y=annual_df["total_volume"],
               name="Volume", marker_color=COLOR_VOL,
               text=annual_df["total_volume"].round(0).astype(int),
               textposition="outside", textfont=dict(size=9)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=annual_df["year"].astype(str), y=annual_df["total_value"],
               name="Value", marker_color=COLOR_VAL,
               text=annual_df["total_value"].round(1),
               textposition="outside", textfont=dict(size=9)),
        row=1, col=2,
    )
    fig.update_layout(
        title="Annual Summary Comparison",
        title_font_size=15,
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(gridcolor="#21262d", tickcolor="#8b949e")
    fig.update_yaxes(gridcolor="#21262d", tickcolor="#8b949e")
    return fig


def chart_mom_growth(df: pd.DataFrame) -> go.Figure:
    df2 = df.dropna(subset=["volume_mom_growth"])
    if df2.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#8b949e"))
        apply_theme(fig)
        return fig
    colors = [COLOR_VAL if v >= 0 else COLOR_ANOMALY for v in df2["volume_mom_growth"]]
    fig = go.Figure(go.Bar(
        x=df2["period"],
        y=df2["volume_mom_growth"],
        marker_color=colors,
        name="MoM Volume Growth %",
    ))
    fig.update_layout(title="Month-over-Month Volume Growth (%)",
                      title_font_size=15, **PLOTLY_LAYOUT)
    fig.add_hline(y=0, line_color="#8b949e", line_width=1, line_dash="dot")
    return fig


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, meta: dict):
    st.markdown(
        """
        <div style='padding:24px 0 8px;'>
          <h1 style='font-size:30px;font-weight:700;color:#e6edf3;margin:0;'>
            India's Digital Payments, Explained Through Data
          </h1>
          <p style='color:#8b949e;font-size:15px;margin-top:8px;'>
            Explore UPI growth, transaction behaviour, and digital payment trends
            through interactive analytics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if meta.get("is_demo"):
        st.markdown(
            "<div class='demo-banner'>⚠️ <strong>DEMO DATA</strong> — "
            "This dashboard is using illustrative synthetic data for testing. "
            "For official figures, upload the NPCI monthly statistics CSV via the sidebar. "
            "Source: <a href='https://www.npci.org.in/product/upi/product-statistics' "
            "target='_blank' style='color:#fbbf24;'>NPCI UPI Statistics</a></div>",
            unsafe_allow_html=True,
        )

    if meta.get("warnings"):
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for w in meta["warnings"]:
                st.warning(w)

    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)
    render_kpi_row(df)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_info1, col_info2 = st.columns([2, 1])
    with col_info1:
        st.markdown("<div class='section-header'>Transaction Volume Overview</div>", unsafe_allow_html=True)
        fig = chart_volume_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    with col_info2:
        st.markdown("<div class='section-header'>Dataset Info</div>", unsafe_allow_html=True)
        annual = get_annual_summary(df)
        st.markdown(
            f"""<div class='info-box'>
            <strong>Reporting Period</strong><br/>{meta.get('date_range','N/A')}<br/><br/>
            <strong>Months of Data</strong><br/>{meta.get('rows_clean', len(df))}<br/><br/>
            <strong>Years Covered</strong><br/>{', '.join(str(y) for y in sorted(df['year'].unique()))}<br/><br/>
            <strong>Source</strong><br/>{meta.get('source','N/A')}<br/><br/>
            <strong>NPCI Reference</strong><br/>
            <a href='https://www.npci.org.in/product/upi/product-statistics'
               target='_blank' style='color:#58a6ff;'>npci.org.in</a>
            </div>""",
            unsafe_allow_html=True,
        )
        if not annual.empty:
            st.markdown("<br/>**Annual Totals**")
            st.dataframe(
                annual[["year", "total_volume", "total_value"]].rename(columns={
                    "year": "Year",
                    "total_volume": "Vol (Cr)",
                    "total_value": "Val (₹LCr)",
                }).style.format({"Vol (Cr)": "{:,.1f}", "Val (₹LCr)": "{:,.2f}"}),
                hide_index=True,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page: Transaction Trends
# ---------------------------------------------------------------------------
def page_trends(df: pd.DataFrame):
    st.markdown("<h2 style='color:#e6edf3;'>📈 Transaction Trends</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Volume Trend",
        "Value Trend",
        "Vol vs Val",
        "YoY Growth",
        "Avg Txn Value",
        "Annual Summary",
    ])

    with tab1:
        st.plotly_chart(chart_volume_trend(df), use_container_width=True)
        st.caption(
            "Transaction Volume is measured in crore (10 million) UPI transactions per month. "
            "Data source: NPCI monthly statistics."
        )

    with tab2:
        st.plotly_chart(chart_value_trend(df), use_container_width=True)
        st.caption(
            "Transaction Value is measured in ₹ lakh crore (₹10 trillion). "
            "This reflects aggregate monetary value of all UPI transfers."
        )

    with tab3:
        st.plotly_chart(chart_vol_vs_val(df), use_container_width=True)
        st.caption(
            "Left axis: Transaction Volume (Crore). "
            "Right axis: Transaction Value (₹ Lakh Crore). "
            "Both axes are independent — this chart shows directional trends, not magnitude comparison."
        )

    with tab4:
        st.plotly_chart(chart_yoy_growth(df), use_container_width=True)
        st.caption(
            "Year-over-Year growth compares each month with the same month in the previous year. "
            "Requires at least 13 months of data. Positive = growth vs prior year."
        )

    with tab5:
        st.plotly_chart(chart_avg_txn_value(df), use_container_width=True)
        st.caption(
            "Average transaction value = Total Value (in ₹) ÷ Total Transactions. "
            "Calculated as: (transaction_value × 1e12) ÷ (transaction_volume × 1e7)."
        )

    with tab6:
        annual = get_annual_summary(df)
        st.plotly_chart(chart_annual_summary(annual), use_container_width=True)
        if not annual.empty:
            st.markdown("**Annual Summary Table**")
            st.dataframe(
                annual.rename(columns={
                    "year": "Year", "total_volume": "Total Vol (Cr)",
                    "total_value": "Total Val (₹LCr)", "avg_monthly_volume": "Avg Monthly Vol",
                    "avg_monthly_value": "Avg Monthly Val", "months_reported": "Months",
                    "yoy_volume_growth": "YoY Vol %", "yoy_value_growth": "YoY Val %",
                }).style.format({
                    "Total Vol (Cr)": "{:,.1f}",
                    "Total Val (₹LCr)": "{:,.2f}",
                    "Avg Monthly Vol": "{:,.1f}",
                    "Avg Monthly Val": "{:,.2f}",
                    "YoY Vol %": "{:+.1f}",
                    "YoY Val %": "{:+.1f}",
                }, na_rep="—"),
                hide_index=True,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page: Payment Insights
# ---------------------------------------------------------------------------
def page_payment_insights(df: pd.DataFrame):
    st.markdown("<h2 style='color:#e6edf3;'>💡 Payment Insights</h2>", unsafe_allow_html=True)
    ts = trend_summary(df)
    anomalies = detect_anomalies(df, "transaction_volume")

    if not ts:
        st.info("Insufficient data for trend analysis. Please check filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Trend overview
        direction = ts.get("direction", "mixed")
        icon = {"rising": "📈", "falling": "📉", "mixed": "↔️"}.get(direction, "↔️")
        st.markdown(
            f"""<div class='insight-card'>
            <div class='insight-title'>📊 Overall Trend</div>
            <p>{icon} Transaction volume is <strong>{direction}</strong> over the selected period.</p>
            <p>Average month-over-month growth: 
               <strong>{ts.get('avg_monthly_volume_growth', float('nan')):.1f}%</strong></p>
            <p>Average month-over-month value growth: 
               <strong>{ts.get('avg_monthly_value_growth', float('nan')):.1f}%</strong></p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Peak & trough
        best = ts.get("best_month", "N/A")
        worst = ts.get("worst_month", "N/A")
        best_vol = ts.get("best_month_volume", float("nan"))
        worst_vol = ts.get("worst_month_volume", float("nan"))
        st.markdown(
            f"""<div class='insight-card'>
            <div class='insight-title'>🏆 Peak & Trough Periods</div>
            <p><strong>Highest Volume:</strong> {best} — 
               {f'{best_vol:,.2f} Cr' if not np.isnan(best_vol) else 'N/A'}</p>
            <p><strong>Lowest Volume:</strong> {worst} — 
               {f'{worst_vol:,.2f} Cr' if not np.isnan(worst_vol) else 'N/A'}</p>
            <p style='font-size:12px;color:#8b949e;'>
            Peak periods may reflect year-end merchant settlements, festive seasons, 
            or government subsidy disbursements. Trough periods may reflect seasonal 
            slowdowns or data reporting gaps. These are observational notes only.
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        # Correlation
        corr = ts.get("correlation_volume_value", float("nan"))
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
        st.markdown(
            f"""<div class='insight-card'>
            <div class='insight-title'>🔗 Volume–Value Correlation</div>
            <p>Pearson correlation between monthly volume and value: 
               <strong>{corr_str}</strong></p>
            <p>{"High correlation (>0.95) means volume and value grew together." 
               if not np.isnan(corr) and corr > 0.95 
               else "Moderate-to-high correlation observed."}</p>
            <p style='font-size:12px;color:#8b949e;'>
            Correlation measures co-movement, not causation. 
            Both metrics rising together may simply reflect platform-level growth.
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Anomaly detection
        st.markdown(
            f"""<div class='insight-card'>
            <div class='insight-title'>⚡ Statistical Anomalies (Volume)</div>
            <p>Periods with MoM change {'>'} 2σ from mean: 
               <strong>{len(anomalies)}</strong></p>
            {"".join([f"<p>• <strong>{r['period']}</strong>: {r['transaction_volume']:,.2f} Cr "
                      f"(z={r['anomaly_z_score']:.1f}σ)</p>" 
                      for _, r in anomalies.iterrows()]) if not anomalies.empty 
             else "<p style='color:#3fb950;'>No unusual periods detected.</p>"}
            <p style='font-size:12px;color:#8b949e;'>
            Statistical anomaly detection uses z-scores. 
            A high z-score means the change was unusual relative to the dataset's own variation.
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """<div class='insight-card'>
        <div class='insight-title'>🗺️ Geographic & Ecosystem Insights</div>
        <p>State-wise UPI transaction data and payment application market share data are 
        <strong>not included</strong> in the current dataset.</p>
        <p>To enable geographic and app-level visualisations, the dataset would need additional columns:</p>
        <ul>
          <li><code>state</code> — State or UT name</li>
          <li><code>payment_app</code> — Payment application (e.g. PhonePe, Google Pay, Paytm)</li>
          <li><code>app_volume_share</code> — Percentage share of total volume</li>
        </ul>
        <p>Official state-level data is available in NPCI's ecosystem statistics reports. 
        If you have such a dataset, upload it via the sidebar and the charts will populate.</p>
        <p style='font-size:12px;color:#8b949e;'>
        Source reference: 
        <a href='https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics' 
           target='_blank' style='color:#58a6ff;'>NPCI UPI Ecosystem Statistics</a>
        </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # MoM growth bar chart
    st.markdown("<div class='section-header' style='margin-top:16px;'>Month-over-Month Volume Growth</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_mom_growth(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Data Explorer
# ---------------------------------------------------------------------------
def page_data_explorer(df: pd.DataFrame, meta: dict):
    st.markdown("<h2 style='color:#e6edf3;'>🔍 Data Explorer</h2>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data loaded. Please upload a CSV or check the demo data file.")
        return

    # Quality summary
    with st.expander("📋 Dataset Quality Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", meta.get("rows_raw", len(df)))
        col2.metric("Clean Rows", meta.get("rows_clean", len(df)))
        col3.metric("Date Range", meta.get("date_range", "N/A"))
        col4.metric("Demo Data", "Yes ⚠️" if meta.get("is_demo") else "No ✅")

        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if missing_cols.empty:
            st.success("✅ No missing values in the loaded dataset.")
        else:
            st.warning(f"Missing values detected: {missing_cols.to_dict()}")

        if meta.get("warnings"):
            st.markdown("**Warnings:**")
            for w in meta["warnings"]:
                st.caption(f"• {w}")

    # Column selector
    available_cols = [c for c in df.columns if c not in ("month_num",)]
    default_cols = [c for c in ["period", "year", "month", "transaction_volume",
                                "transaction_value", "avg_transaction_value",
                                "volume_mom_growth", "volume_yoy_growth"] if c in available_cols]
    selected_cols = st.multiselect("Select columns to display", available_cols, default=default_cols)

    # Search
    search_term = st.text_input("🔍 Search (filters period column)", "")

    display_df = df[selected_cols].copy() if selected_cols else df.copy()

    if search_term:
        mask = display_df.apply(lambda col: col.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]

    # Formatting
    format_dict = {}
    if "transaction_volume" in display_df.columns:
        format_dict["transaction_volume"] = "{:,.2f}"
    if "transaction_value" in display_df.columns:
        format_dict["transaction_value"] = "{:,.3f}"
    if "avg_transaction_value" in display_df.columns:
        format_dict["avg_transaction_value"] = "{:,.0f}"
    if "volume_mom_growth" in display_df.columns:
        format_dict["volume_mom_growth"] = "{:+.2f}"
    if "volume_yoy_growth" in display_df.columns:
        format_dict["volume_yoy_growth"] = "{:+.2f}"

    st.markdown(f"**Showing {len(display_df)} rows**")
    st.dataframe(
        display_df.style.format(format_dict, na_rep="—"),
        use_container_width=True,
        height=400,
    )

    # CSV download
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_bytes,
        file_name="upi_pulse_filtered_data.csv",
        mime="text/csv",
    )

    st.caption(
        "Units: transaction_volume in Crore | transaction_value in ₹ Lakh Crore | "
        "avg_transaction_value in ₹ | growth rates in %"
    )


# ---------------------------------------------------------------------------
# Page: AI Insights
# ---------------------------------------------------------------------------
def page_ai_insights(df: pd.DataFrame):
    st.markdown("<h2 style='color:#e6edf3;'>🤖 AI Insights</h2>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-box'>All numerical statements in this section are derived from "
        "the currently loaded dataset. Interpretations are labelled as AI-assisted insights — "
        "they are NOT official statements from NPCI or RBI.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Please upload a dataset or check filters.")
        return

    annual = get_annual_summary(df)

    with st.spinner("Generating insights…"):
        local_insights = generate_local_insights(df, annual)
        insights = enhance_with_llm(local_insights, df)

    ai_badge = "🤖 AI-Enhanced" if insights.get("ai_enhanced") else "📐 Rule-Based"

    sections = [
        ("📊 KPI Summary", "kpi_summary"),
        ("📈 Trend Analysis", "trend_narrative"),
        ("📉 Growth Analysis", "growth_analysis"),
        ("⚡ Anomaly Detection", "anomaly_detection"),
        ("❓ Suggested Questions", "further_questions"),
    ]

    for title, key in sections:
        content = insights.get(key, "")
        if content:
            st.markdown(
                f"""<div class='insight-card'>
                <div class='insight-title'>{title}</div>
                {content.replace(chr(10), '<br/>')}
                </div>""",
                unsafe_allow_html=True,
            )

    # Executive summary
    exec_summary = insights.get("executive_summary", "")
    if exec_summary:
        st.markdown(
            f"""<div class='insight-card' style='border-color:#388bfd;'>
            <div class='insight-title' style='color:#388bfd;'>
              📄 Executive Summary &nbsp;<span style='color:#8b949e;font-size:10px;'>{ai_badge}</span>
            </div>
            {exec_summary.replace(chr(10), '<br/>')}
            </div>""",
            unsafe_allow_html=True,
        )

    # Forecast section
    st.markdown("---")
    st.markdown("<div class='section-header'>📡 Trend Forecast (Baseline Linear Model)</div>",
                unsafe_allow_html=True)

    forecast_df = simple_forecast(df, periods_ahead=6, metric="transaction_volume")
    if forecast_df is not None:
        # Combine history + forecast for display
        history = df[["period", "transaction_volume"]].copy()
        history["type"] = "Historical"
        history = history.rename(columns={"transaction_volume": "value"})

        fcast = forecast_df[["period", "forecast"]].copy()
        fcast = fcast.rename(columns={"forecast": "value"})
        fcast["type"] = "Forecast"

        combined = pd.concat([history, fcast], ignore_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["period"], y=history["value"],
            name="Historical", line=dict(color=COLOR_VOL, width=2),
            mode="lines+markers", marker_size=3,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["period"], y=forecast_df["forecast"],
            name="Forecast", line=dict(color=COLOR_GROWTH, width=2, dash="dash"),
            mode="lines+markers", marker_size=5,
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["period"], forecast_df["period"].iloc[::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,166,87,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Uncertainty Band",
            showlegend=True,
        ))
        apply_theme(fig)
        fig.update_layout(title="Transaction Volume Forecast — Linear Trend Baseline (Illustrative)",
                          title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "⚠️ **Forecast Disclaimer:** This is a simple linear trend extrapolation trained "
            "on historical data only. It does not model seasonality, policy changes, or external "
            "events. Uncertainty bands show ±1 standard deviation of residuals — NOT prediction "
            "intervals. This forecast is for educational illustration only and must not be "
            "interpreted as a reliable prediction of future UPI volumes."
        )
        st.caption(f"Model: {forecast_df['model'].iloc[0]}")
    else:
        st.info(
            "Forecasting is disabled for the current selection. "
            "At least 24 months of continuous data are required to generate a trend baseline. "
            "Adjust your filters to include more historical months."
        )


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about():
    st.markdown("<h2 style='color:#e6edf3;'>ℹ️ About This Project</h2>", unsafe_allow_html=True)

    content = {
        "🎯 Project Title & Objective": """
**UPI Pulse India: Digital Payments and UPI Transaction Trend Analysis in India**

This project was built for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**.
The objective is to analyse India's Unified Payments Interface (UPI) transaction ecosystem through
exploratory data analytics, interactive visualisations, and AI-assisted interpretations.
""",
        "📋 Problem Statement": """
India's digital payments ecosystem has grown exponentially since the launch of UPI in April 2016.
However, raw transaction statistics from official sources are often difficult to interpret without 
visualisation and contextualisation.

This project addresses: *How can publicly available UPI transaction data be transformed into 
actionable insights through data analytics and interactive dashboards?*
""",
        "📂 Dataset Source & Attribution": """
- **Primary Source:** National Payments Corporation of India (NPCI)
  - UPI Product Statistics: https://www.npci.org.in/product/upi/product-statistics
  - UPI Ecosystem Statistics: https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics
- **Data Format:** Monthly aggregate UPI statistics (volume in crore, value in ₹ lakh crore)
- **Coverage:** April 2016 to present (official) | April 2016 – December 2024 (demo dataset)
- **Demo Data:** The bundled `upi_transactions_sample.csv` contains **illustrative synthetic data** 
  for testing only. It is clearly marked as DEMO DATA and should never be presented as official statistics.
""",
        "🛠️ Technologies Used": """
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥1.32 | Web dashboard |
| Pandas | ≥2.0 | Data processing |
| NumPy | ≥1.24 | Numerical calculations |
| Plotly | ≥5.18 | Interactive charts |
| Scikit-learn | ≥1.3 | Trend forecasting |
| python-dotenv | ≥1.0 | API key management |
""",
        "🔬 Methodology": """
1. **Data Ingestion** — Load CSV from upload, official file, or demo file with validation.
2. **Data Cleaning** — Handle missing values, duplicates, invalid entries, and unit normalisation.
3. **KPI Computation** — Calculate totals, averages, growth rates using unit-safe formulas.
4. **Trend Analysis** — Monthly and yearly growth rates, anomaly detection via z-scores.
5. **Visualisation** — Interactive Plotly charts rendered in a Streamlit dashboard.
6. **AI Insights** — Rule-based natural language summaries; optional LLM enhancement.
7. **Forecasting** — Simple linear baseline (when ≥24 months of data are available).
""",
        "❓ Key Analytical Questions": """
1. How has UPI transaction volume grown since launch (April 2016)?
2. What is the compound annual growth rate of UPI transactions?
3. How did COVID-19 lockdowns (Apr–May 2020) affect transaction volumes?
4. Has the average transaction value increased or decreased over time?
5. Which months or quarters show the strongest seasonal patterns?
6. What is the correlation between transaction volume and value growth?
""",
        "⚠️ Limitations": """
- The demo dataset contains illustrative figures — it is NOT certified by NPCI or RBI.
- Official data must be manually downloaded from the NPCI website as no stable public API exists.
- State-level and payment-app-level breakdowns are not included in the current dataset.
- The linear forecast model is extremely naive and should not be used for business decisions.
- This dashboard is an **educational data analytics project** only. It is not a banking, 
  investment, or payment-processing service.
""",
        "▶️ Reproduction Instructions": """
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/upi-pulse-india.git
cd upi-pulse-india

# 2. Create virtual environment
python -m venv venv
venv\\Scripts\\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set API key for AI enhancement
copy .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

# 5. Run the dashboard
streamlit run app.py

# 6. Run the test suite
python -m pytest tests/ -v
```

Upload official NPCI data via the sidebar to replace the demo dataset.
""",
    }

    for title, body in content.items():
        st.markdown(
            f"""<div class='about-section'>
            <h3>{title}</h3>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(body)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

    st.markdown(
        "<p style='color:#8b949e;font-size:12px;text-align:center;'>"
        "This is an educational data analytics project created for the "
        "AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026. "
        "Author: Mo Faishal Haque"
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    # Sidebar (returns page + filters + upload)
    # We need a temp render to get the uploaded file first
    with st.sidebar:
        uploaded_temp = st.file_uploader(
            "Upload official CSV",
            type=["csv"],
            help="Upload NPCI monthly statistics CSV.",
            key="_upload_check",
        )

    @st.cache_data(show_spinner="Loading data…")
    def load_cached(upload_name):
        return load_data()

    if uploaded_temp is not None:
        df_raw, meta = load_data(uploaded_temp)
    else:
        df_raw, meta = load_cached("default")

    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 16px 0 8px;'>
              <div style='font-size:26px;'>💳</div>
              <div style='font-size:18px; font-weight:700; color:#58a6ff;'>UPI Pulse India</div>
              <div style='font-size:11px; color:#8b949e; margin-top:2px;'>
                Digital Payments Analytics
              </div>
            </div>
            <hr style='border-color:#21262d; margin:10px 0 16px;'/>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "📊 Overview",
                "📈 Transaction Trends",
                "💡 Payment Insights",
                "🔍 Data Explorer",
                "🤖 AI Insights",
                "ℹ️ About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#21262d; margin:16px 0;'/>", unsafe_allow_html=True)

        selected_years = []
        start_period = None
        end_period = None

        if not df_raw.empty:
            st.markdown(
                "<div style='font-size:12px;color:#8b949e;text-transform:uppercase;"
                "letter-spacing:0.08em;margin-bottom:8px;'>Filters</div>",
                unsafe_allow_html=True,
            )
            available_years = sorted(df_raw["year"].unique().tolist())
            selected_years = st.multiselect(
                "Year(s)", available_years, default=available_years
            )
            if not selected_years:
                selected_years = available_years

            periods = df_raw["period"].tolist()
            c1, c2 = st.columns(2)
            with c1:
                start_period = st.selectbox("From", periods, index=0)
            with c2:
                end_period = st.selectbox("To", periods, index=len(periods) - 1)

        st.markdown(
            f"<div style='font-size:11px;color:#8b949e;margin-top:8px;'>"
            f"{'⚠️ DEMO DATA' if meta.get('is_demo') else '✅ Dataset loaded'}<br/>"
            f"{meta.get('date_range','')}</div>",
            unsafe_allow_html=True,
        )

    # Apply filters
    df = filter_data(df_raw, years=selected_years, start_period=start_period, end_period=end_period)
    meta["rows_clean"] = len(df)
    if not df.empty:
        from src.data_loader import _date_range_str
        meta["date_range"] = _date_range_str(df)

    # Render selected page
    if page == "📊 Overview":
        page_overview(df, meta)
    elif page == "📈 Transaction Trends":
        page_trends(df)
    elif page == "💡 Payment Insights":
        page_payment_insights(df)
    elif page == "🔍 Data Explorer":
        page_data_explorer(df, meta)
    elif page == "🤖 AI Insights":
        page_ai_insights(df)
    elif page == "ℹ️ About Project":
        page_about()


if __name__ == "__main__":
    main()
