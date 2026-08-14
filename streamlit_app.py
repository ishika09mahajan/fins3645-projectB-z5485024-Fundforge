"""FundForge - Part B dashboard (Station 4).

A deployed decision tool that READS the precomputed results/ artifacts produced by
scripts/run_part_b.py. It never runs VADER or a backtest at request time, so it is
fast and free-tier friendly. Everything shown here is reproducible from the CSVs in
results/, which are built offline and committed to the repo.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to a public GitHub repo, then connect it on share.streamlit.io
               with entrypoint streamlit_app.py (see brief App. D).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

FAM_COLOR = {"equity": "#2563eb", "crypto": "#f59e0b", "combined": "#10b981"}
BASE_C, TILT_C = "#94a3b8", "#7c3aed"

st.set_page_config(page_title="FundForge", page_icon="🔥", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1250px;}
  .ff-hero {background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #7c3aed 100%);
            padding: 1.6rem 1.9rem; border-radius: 16px; color: #f8fafc; margin-bottom: 1.2rem;}
  .ff-hero h1 {font-size: 2.0rem; margin: 0; font-weight: 800; letter-spacing: -0.5px;}
  .ff-hero p  {margin: .35rem 0 0; opacity: .85; font-size: .98rem;}
  .ff-card {background: var(--secondary-background-color); border-radius: 14px;
            padding: 1.0rem 1.1rem; border: 1px solid rgba(148,163,184,.18); height: 100%;}
  .ff-card .k {font-size: .78rem; text-transform: uppercase; letter-spacing: .04em;
               opacity: .65; margin-bottom: .25rem;}
  .ff-card .v {font-size: 1.55rem; font-weight: 750; line-height: 1.1;}
  .ff-card .s {font-size: .82rem; opacity: .7; margin-top: .2rem;}
  .stTabs [data-baseweb="tab-list"] {gap: .3rem;}
  .stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- loaders
@st.cache_data
def load_csv(path: pathlib.Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def fam_of(name: str) -> str:
    return name.split("_")[0]


def card(col, k, v, s=""):
    col.markdown(f'<div class="ff-card"><div class="k">{k}</div>'
                 f'<div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def tilt_weights(weights: pd.Series, signal: pd.Series, strength: float) -> pd.Series:
    """exp-tilt a weight vector by a per-asset signal, renormalised (the app blender)."""
    sig = signal.reindex(weights.index).fillna(0.0)
    w = weights * np.exp(strength * sig)
    return w / w.sum() if w.sum() > 0 else weights


# --------------------------------------------------------------------------- load all
perf = load_csv(TABLES / "performance_metrics.csv")
fret = load_csv(DATA / "fund_returns.csv", index_col=0, parse_dates=True)
fwt = load_csv(DATA / "fund_weights.csv", index_col=0)
fg = load_csv(DATA / "sector_fear_greed_100.csv", index_col=0, parse_dates=True)
zsig = load_csv(DATA / "sector_sentiment_index.csv", index_col=0, parse_dates=True)
fus = load_csv(TABLES / "fusion_metrics.csv")
fusr = load_csv(DATA / "fusion_returns.csv", index_col=0, parse_dates=True)
tk_sec = load_csv(DATA / "ticker_sector.csv")
TK2SEC = dict(zip(tk_sec["ticker"], tk_sec["sector"]))

# --------------------------------------------------------------------------- hero
st.markdown('<div class="ff-hero"><h1>🔥 FundForge</h1>'
            '<p>Four portfolio engines across equities, crypto and a blended universe, '
            'with a look-ahead-safe news-sentiment overlay. Built on a walk-forward '
            'out-of-sample backtest.</p></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
fam_sel = st.sidebar.multiselect("Fund families", ["equity", "crypto", "combined"],
                                 default=["equity", "crypto", "combined"])
sectors = list(fg.columns)
sector_sel = st.sidebar.selectbox("Sentiment sector (gauge)", sectors,
                                  index=sectors.index("Tech") if "Tech" in sectors else 0)
st.sidebar.caption("All figures are out-of-sample and reproducible from results/ (no "
                   "look-ahead: weights use only trailing data, sentiment is lagged).")

perf_f = perf[perf["fund"].apply(fam_of).isin(fam_sel)] if fam_sel else perf

tabs = st.tabs(["Overview", "Funds", "Sentiment", "Sentiment overlay",
                "Allocation blender", "Data"])

# =========================================================== Overview
with tabs[0]:
    best = perf_f.loc[perf_f["sharpe"].idxmax()] if len(perf_f) else perf.loc[perf["sharpe"].idxmax()]
    topret = perf_f.loc[perf_f["ann_return"].idxmax()] if len(perf_f) else perf.loc[perf["ann_return"].idxmax()]
    c = st.columns(4)
    card(c[0], "Funds", f"{len(perf_f)}", "portfolio x universe")
    card(c[1], "Best Sharpe", f"{best['sharpe']:.2f}", best["fund"])
    card(c[2], "Top annual return", f"{topret['ann_return']*100:.1f}%", topret["fund"])
    card(c[3], "Sectors tracked", f"{len(sectors)}", "daily news sentiment")

    st.subheader("Risk and return")
    d = perf_f.copy()
    d["family"] = d["fund"].apply(fam_of)
    fig = px.scatter(d, x="ann_vol", y="ann_return", color="family",
                     hover_name="fund",
                     hover_data={"sharpe": ":.2f", "max_drawdown": ":.2%",
                                 "ann_vol": ":.2%", "ann_return": ":.2%", "family": False},
                     color_discrete_map=FAM_COLOR, height=520,
                     labels={"ann_vol": "annualised volatility", "ann_return": "annualised return"})
    fig.update_traces(marker=dict(size=13))
    fig.update_traces(textposition="top center", textfont_size=9, marker=dict(size=13))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Performance table")
    st.dataframe(perf_f.set_index("fund").style.format({
        "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
        "max_drawdown": "{:.2%}", "avg_turnover": "{:.3f}"}), width="stretch")

# =========================================================== Funds
with tabs[1]:
    st.subheader("Growth of $1 (out-of-sample)")
    cols = [c for c in fret.columns if fam_of(c) in fam_sel]
    growth = (1 + fret[cols]).cumprod()
    fig = go.Figure()
    for c in cols:
        g = growth[c].dropna()
        fig.add_trace(go.Scatter(x=g.index, y=g.values, name=c, mode="lines",
                                 line=dict(color=FAM_COLOR[fam_of(c)], width=1.3),
                                 opacity=0.9))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="growth of $1", legend=dict(font=dict(size=9)))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Current holdings")
    fund_pick = st.selectbox("Fund", [c for c in fwt.columns if fam_of(c) in fam_sel] or list(fwt.columns))
    w = fwt[fund_pick].dropna().sort_values(ascending=False)
    w = w[w.abs() > 1e-6]
    figw = px.bar(x=w.values, y=w.index, orientation="h", height=max(320, 22 * len(w)),
                  labels={"x": "weight", "y": ""})
    figw.update_traces(marker_color="#2563eb")
    figw.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figw, width="stretch")

# =========================================================== Sentiment
with tabs[2]:
    latest = fg[sector_sel].dropna()
    val = float(latest.iloc[-1]) if len(latest) else 50.0
    label = ("Extreme fear" if val < 25 else "Fear" if val < 45 else
             "Neutral" if val < 55 else "Greed" if val < 75 else "Extreme greed")
    cc = st.columns([1, 1.4])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={"text": f"{sector_sel} - Fear & Greed"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#7c3aed"},
               "steps": [{"range": [0, 25], "color": "#fecaca"},
                         {"range": [25, 45], "color": "#fed7aa"},
                         {"range": [45, 55], "color": "#e5e7eb"},
                         {"range": [55, 75], "color": "#bbf7d0"},
                         {"range": [75, 100], "color": "#86efac"}]}))
    gauge.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    cc[0].plotly_chart(gauge, width="stretch")
    cc[1].markdown(f"### {label}")
    cc[1].markdown(f"Latest raw sentiment for **{sector_sel}** is **{val:.1f} / 100** "
                   f"as of {latest.index[-1].date() if len(latest) else 'n/a'}. Raw scores "
                   "sit above 50 on most days, so the tradable signal is the standardised "
                   "series below (expanding window, mean 0).")

    st.subheader("Raw Fear & Greed by sector (21-day smoothed)")
    sm = fg.rolling(21, min_periods=1).mean()
    st.line_chart(sm, height=320)

    st.subheader("Standardised sentiment signal (look-ahead safe)")
    st.line_chart(zsig, height=320)

# =========================================================== Overlay
with tabs[3]:
    st.subheader("Does news sentiment add value?")
    st.markdown("Each equity/combined fund is run with and without a lagged sentiment "
                "tilt. The overlay **helps the return-seeking max-Sharpe funds** and "
                "**reduces drawdown in most funds**, while slightly hurting the "
                "risk-minimising funds. A weak but real, defensive signal, not a free lunch.")
    m = fus.melt(id_vars="fund", value_vars=["base_sharpe", "tilt_sharpe"],
                 var_name="kind", value_name="sharpe")
    fig = px.bar(m, x="fund", y="sharpe", color="kind", barmode="group", height=430,
                 color_discrete_map={"base_sharpe": BASE_C, "tilt_sharpe": TILT_C})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")

    md = fus.melt(id_vars="fund", value_vars=["base_max_dd", "tilt_max_dd"],
                  var_name="kind", value_name="max_dd")
    fig2 = px.bar(md, x="fund", y="max_dd", color="kind", barmode="group", height=430,
                  color_discrete_map={"base_max_dd": BASE_C, "tilt_max_dd": TILT_C})
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-35)
    st.plotly_chart(fig2, width="stretch")

    fund_o = st.selectbox("Overlay equity curve", fus["fund"].tolist(),
                          index=fus["fund"].tolist().index("combined_max_sharpe")
                          if "combined_max_sharpe" in fus["fund"].tolist() else 0)
    b, t = f"{fund_o}__base", f"{fund_o}__tilt"
    if b in fusr.columns:
        gb = (1 + fusr[b].dropna()).cumprod()
        gt = (1 + fusr[t].dropna()).cumprod()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=gb.index, y=gb.values, name="base", line=dict(color=BASE_C)))
        fig3.add_trace(go.Scatter(x=gt.index, y=gt.values, name="+ sentiment tilt",
                                  line=dict(color=TILT_C)))
        fig3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="growth of $1")
        st.plotly_chart(fig3, width="stretch")

# =========================================================== Blender
with tabs[4]:
    st.subheader("Allocation blender")
    st.markdown("Take a fund's current holdings and tilt them by each stock's **latest "
                "sector sentiment**. Slide the strength to see conviction move between "
                "names. This is the same exp-tilt rule the backtest applies at each "
                "rebalance (`fusion.apply_sentiment`).")
    eq_funds = [c for c in fwt.columns if fam_of(c) in ("equity", "combined")]
    fund_b = st.selectbox("Fund to blend", eq_funds or list(fwt.columns))
    strength = st.slider("Sentiment tilt strength", 0.0, 1.5, 0.5, 0.05)

    base_w = fwt[fund_b].dropna()
    base_w = base_w[base_w.abs() > 1e-6]
    latest_sig = zsig.iloc[-1]                       # latest standardised sector signal
    tk_sig = pd.Series({tk: latest_sig.get(TK2SEC.get(tk), 0.0) for tk in base_w.index})
    new_w = tilt_weights(base_w, tk_sig, strength)

    comp = pd.DataFrame({"base": base_w, "tilted": new_w})
    comp = comp.reindex(comp["tilted"].sort_values(ascending=False).index)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp.index, x=comp["base"], name="base", orientation="h",
                         marker_color=BASE_C))
    fig.add_trace(go.Bar(y=comp.index, x=comp["tilted"], name="tilted", orientation="h",
                         marker_color=TILT_C))
    fig.update_layout(barmode="group", height=max(340, 26 * len(comp)),
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="weight")
    st.plotly_chart(fig, width="stretch")
    moved = (new_w - base_w.reindex(new_w.index).fillna(0)).abs().sum() / 2
    st.caption(f"Turnover from the tilt at strength {strength:.2f}: {moved:.1%} of the book.")

# =========================================================== Data
with tabs[5]:
    st.subheader("Hosted data source")
    st.caption("Loaded live from the course data source via src.data_access, cached "
               "daily. The rest of the dashboard runs on precomputed results/ so it "
               "stays fast on the free tier.")
    try:
        from src import data_access

        @st.cache_data(ttl=86_400, show_spinner="Loading hosted data...")
        def _hosted_equities():
            return data_access.load_equity_prices()

        eq = _hosted_equities()
        st.write(f"Equity prices: {eq.shape[0]:,} rows, {eq['ticker'].nunique()} tickers, "
                 f"{len(eq.columns)} columns.")
        st.dataframe(eq.head(20), width="stretch")
    except Exception as exc:
        st.info("The live data source is not reachable in this environment, so the "
                "dashboard is running entirely on the precomputed results/ artifacts. "
                f"({type(exc).__name__})")

st.caption("FundForge - FINS3645 Part B. Data is out-of-sample and precomputed; the app "
           "reads results/ only. Sentiment is VADER tuned with a finance lexicon; the "
           "overlay is lagged and standardised on an expanding window (no look-ahead).")