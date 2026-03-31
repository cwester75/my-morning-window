#!/usr/bin/env python3
"""
Morning Window — Streamlit Cloud App
======================================
Mobile-responsive daily market dashboard.

Local:    streamlit run streamlit_app.py
Deploy:   Push to GitHub → connect repo at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, datetime
from scipy.stats import percentileofscore

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Morning Window",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

TICKERS = ["MSFT", "AAPL", "AMZN", "META", "GOOGL", "NFLX", "TSLA", "NVDA"]
MACRO   = {"$USD": "DX=F", "GOLD": "GC=F", "NSDQ 100": "QQQ",
           "10Y Yield": "^TNX", "TLT": "TLT"}
SPX     = "^GSPC"
FACTORS = {"Momentum": "MTUM", "Quality": "QUAL", "Value": "VLUE",
           "Low Vol": "USMV", "High Beta": "SPHB", "Dividend": "SCHD",
           "Equal Wt": "RSP", "Benchmark": "SPY"}

DAYS_1M, DAYS_3M, DAYS_6M = 21, 63, 126

CACHE_TTL = 300  # 5 minutes


# ═══════════════════════════════════════════════════════════════
# DATA LAYER (cached)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_close(sym, period="1y"):
    return (yf.Ticker(sym)
              .history(period=period, interval="1d", auto_adjust=False)["Close"]
              .tz_localize(None))


def pct_ago(series, n):
    idx = max(0, len(series) - 1 - n)
    return (series.iloc[-1] / series.iloc[idx] - 1) * 100


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_fundamentals():
    today = pd.Timestamp(date.today()).normalize()
    y_start = pd.Timestamp(today.year, 1, 1)
    rows = []
    for sym in TICKERS:
        try:
            h = fetch_close(sym)
            if len(h) < DAYS_6M + 1:
                continue
            last, prev = h.iloc[-1], h.iloc[-2]
            h_y = h[h.index >= y_start]
            rows.append({
                "Ticker": sym,
                "Price": round(last, 2),
                "1D %": round((last / prev - 1) * 100, 2),
                "1M %": round(pct_ago(h, DAYS_1M), 2),
                "3M %": round(pct_ago(h, DAYS_3M), 2),
                "YTD %": round((last / h_y.iloc[0] - 1) * 100, 2) if len(h_y) > 0 else None,
                "1Y %": round(pct_ago(h, len(h) - 1), 2),
                "vs 52wk Hi %": round((last / h.max() - 1) * 100, 2),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_relative():
    today = pd.Timestamp(date.today()).normalize()
    y_start = pd.Timestamp(today.year, 1, 1)
    base = fetch_close(SPX)
    base_last = base.iloc[-1]
    base_rets = {
        "1D": (base_last / base.iloc[-2] - 1) * 100,
        "1M": pct_ago(base, DAYS_1M),
        "3M": pct_ago(base, DAYS_3M),
        "YTD": (base_last / base[base.index >= y_start].iloc[0] - 1) * 100,
    }
    rows = []
    for sym in TICKERS:
        try:
            h = fetch_close(sym)
            if len(h) < DAYS_3M + 1:
                continue
            last = h.iloc[-1]
            h_y = h[h.index >= y_start]
            raw = {
                "1D": (last / h.iloc[-2] - 1) * 100,
                "1M": pct_ago(h, DAYS_1M),
                "3M": pct_ago(h, DAYS_3M),
                "YTD": (last / h_y.iloc[0] - 1) * 100 if len(h_y) > 0 else 0,
            }
            rows.append({
                "Ticker": sym,
                **{f"Rel {k} %": round(raw[k] - base_rets[k], 2) for k in raw}
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_volume():
    rows = []
    for sym in TICKERS:
        try:
            vol = (yf.Ticker(sym)
                     .history(period="1y", interval="1d", auto_adjust=False)["Volume"]
                     .tz_localize(None))
            if len(vol) < DAYS_6M + 1:
                continue
            v1d = vol.iloc[-1]
            a30 = vol.iloc[-30:].mean()
            rows.append({
                "Ticker": sym,
                "Vol 1D": int(v1d),
                "30D Avg": int(a30),
                "vs 30D %": round((v1d / a30 - 1) * 100, 1),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_volatility():
    rows = []
    for sym in TICKERS:
        try:
            h = fetch_close(sym, period="5y")
            ret = h.pct_change().dropna()
            rv = ret.rolling(21).std() * np.sqrt(252)
            if len(rv.dropna()) < DAYS_6M:
                continue
            cur = rv.iloc[-1]
            rv_series = rv.dropna()
            rows.append({
                "Ticker": sym,
                "RV (21d)": round(cur * 100, 2),
                "RV %ile": round(percentileofscore(rv_series, cur), 0),
                "RV 1M Chg %": round((cur / rv.shift(21).iloc[-1] - 1) * 100, 1),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_corr_snapshot():
    all_series = {SPX: fetch_close(SPX)}
    for name, sym in MACRO.items():
        try:
            all_series[name] = fetch_close(sym)
        except Exception:
            continue
    df = pd.DataFrame(all_series).dropna()
    rets = df.pct_change().dropna()
    rows = []
    for name in MACRO:
        if name not in rets.columns:
            continue
        rows.append({
            "Metric": name,
            "15D": round(rets[name].iloc[-15:].corr(rets[SPX].iloc[-15:]), 2),
            "30D": round(rets[name].iloc[-30:].corr(rets[SPX].iloc[-30:]), 2),
            "90D": round(rets[name].iloc[-90:].corr(rets[SPX].iloc[-90:]), 2),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_corr_chart_data():
    all_series = {"SPX": fetch_close(SPX)}
    for name, sym in MACRO.items():
        try:
            all_series[name] = fetch_close(sym)
        except Exception:
            continue
    df = pd.DataFrame(all_series).dropna()
    rets = df.pct_change().dropna()
    result = {}
    for name in MACRO:
        if name not in rets.columns:
            continue
        corr = rets["SPX"].rolling(30).corr(rets[name]).dropna()
        result[name] = corr
    return result


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_factors():
    try:
        px = yf.download(list(FACTORS.values()), auto_adjust=True,
                         progress=False)["Close"].dropna(how="all")
        px = px.rename(columns={v: k for k, v in FACTORS.items()})
    except Exception:
        return pd.DataFrame()
    rows = []
    for name in FACTORS:
        if name not in px.columns:
            continue
        s = px[name].dropna()
        r1m = (s.iloc[-1] / s.iloc[-22] - 1) * 100 if len(s) > 22 else None
        r12m = (s.iloc[-1] / s.iloc[-253] - 1) * 100 if len(s) > 253 else None
        rows.append({
            "Factor": name,
            "1M %": round(r1m, 2) if r1m else None,
            "12M %": round(r12m, 2) if r12m else None,
        })
    return pd.DataFrame(rows).sort_values("12M %", ascending=False)


# ═══════════════════════════════════════════════════════════════
# STYLING HELPERS
# ═══════════════════════════════════════════════════════════════

def color_negative_red(val):
    """Style: green for positive, red for negative."""
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    return "color: #4ade80" if val >= 0 else "color: #f87171"


def style_dataframe(df, pct_cols=None):
    """Apply green/red coloring to percentage columns."""
    if pct_cols is None:
        pct_cols = [c for c in df.columns if "%" in c]
    styler = df.style.format(precision=2, na_rep="—")
    for col in pct_cols:
        if col in df.columns:
            styler = styler.map(color_negative_red, subset=[col])
    return styler


# ═══════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>Morning Window</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align:center; color:#888; margin-top:0;'>"
    f"{datetime.now().strftime('%A, %B %d, %Y  %I:%M %p')}"
    f"</p>",
    unsafe_allow_html=True
)

# Refresh button
col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
with col_r2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Load data with progress ──
with st.spinner("Loading market data..."):
    df_fund   = build_fundamentals()
    df_rel    = build_relative()
    df_vol    = build_volume()
    df_rv     = build_volatility()
    df_corr   = build_corr_snapshot()
    corr_data = build_corr_chart_data()
    df_factor = build_factors()

# ── Navigation tabs (mobile-friendly) ──
tabs = st.tabs([
    "Returns",
    "Volume",
    "Volatility",
    "Correlations",
    "Factors",
])

# ── Tab 1: Returns ──
with tabs[0]:
    st.subheader("Absolute Returns")
    if not df_fund.empty:
        st.dataframe(
            style_dataframe(df_fund),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    st.subheader("Relative vs S&P 500")
    if not df_rel.empty:
        st.dataframe(
            style_dataframe(df_rel),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

# ── Tab 2: Volume ──
with tabs[1]:
    st.subheader("Volume Snapshot")
    if not df_vol.empty:
        st.dataframe(
            style_dataframe(df_vol, pct_cols=["vs 30D %"]),
            use_container_width=True,
            hide_index=True,
            height=340,
            column_config={
                "Vol 1D": st.column_config.NumberColumn(format="%d"),
                "30D Avg": st.column_config.NumberColumn(format="%d"),
            },
        )

    # Volume bar chart
    if not df_vol.empty:
        st.subheader("Today vs 30D Average")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df_vol["Ticker"], y=df_vol["vs 30D %"],
            marker_color=[
                "#4ade80" if v >= 0 else "#f87171"
                for v in df_vol["vs 30D %"]
            ],
        ))
        fig_vol.update_layout(
            yaxis_title="% vs 30D Avg",
            template="plotly_dark",
            height=300,
            margin=dict(t=10, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ── Tab 3: Volatility ──
with tabs[2]:
    st.subheader("Realized Volatility (21-Day)")
    if not df_rv.empty:
        st.dataframe(
            style_dataframe(df_rv, pct_cols=["RV 1M Chg %"]),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    # Percentile bar chart
    if not df_rv.empty:
        st.subheader("RV Percentile (5Y History)")
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Bar(
            x=df_rv["Ticker"], y=df_rv["RV %ile"],
            marker_color=[
                "#f87171" if v >= 80 else "#facc15" if v >= 50 else "#4ade80"
                for v in df_rv["RV %ile"]
            ],
        ))
        fig_pct.update_layout(
            yaxis_title="Percentile",
            yaxis_range=[0, 100],
            template="plotly_dark",
            height=300,
            margin=dict(t=10, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

# ── Tab 4: Correlations ──
with tabs[3]:
    st.subheader("Correlations vs S&P 500")
    if not df_corr.empty:
        st.dataframe(
            style_dataframe(df_corr, pct_cols=["15D", "30D", "90D"]),
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    # Rolling correlation chart
    if corr_data:
        st.subheader("Rolling 30D Correlation")
        fig_corr = go.Figure()
        for name, series in corr_data.items():
            fig_corr.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=name, mode="lines",
            ))
        fig_corr.update_layout(
            yaxis_title="Correlation",
            yaxis_range=[-1, 1],
            template="plotly_dark",
            height=350,
            margin=dict(t=10, b=30, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0),
        )
        fig_corr.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 5: Factors ──
with tabs[4]:
    st.subheader("Factor League Table")
    if not df_factor.empty:
        st.dataframe(
            style_dataframe(df_factor),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

    # Factor bar chart
    if not df_factor.empty:
        st.subheader("12-Month Factor Returns")
        fig_fac = go.Figure()
        fig_fac.add_trace(go.Bar(
            x=df_factor["Factor"],
            y=df_factor["12M %"],
            marker_color=[
                "#4ade80" if v and v >= 0 else "#f87171"
                for v in df_factor["12M %"]
            ],
        ))
        fig_fac.update_layout(
            yaxis_title="12M Return %",
            template="plotly_dark",
            height=300,
            margin=dict(t=10, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_fac, use_container_width=True)

# ── Footer ──
st.divider()
st.markdown(
    "<p style='text-align:center; color:#555; font-size:12px;'>"
    "Auto-refreshes every 5 min via cache TTL. "
    "Tap <b>Refresh Data</b> for immediate update. "
    "Add to home screen for app-like experience."
    "</p>",
    unsafe_allow_html=True
)
