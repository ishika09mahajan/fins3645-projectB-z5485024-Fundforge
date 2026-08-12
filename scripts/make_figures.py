"""Generate the Part B figures from the saved results/ CSVs.

    python scripts/make_figures.py

Reads results/data and results/tables (produced by run_part_b.py) and writes
PNGs to results/figures. The sentiment-coverage figure re-scores the headlines
with base VADER vs the finance-tuned analyzer to show the innovation's effect.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- house style ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
FAM_COLOR = {"equity": "#2563eb", "crypto": "#f59e0b", "combined": "#10b981"}
BASE_C, TILT_C = "#94a3b8", "#6d28d9"


def _fam(name): return name.split("_")[0]


def fig_equity_curves():
    r = pd.read_csv(DATA / "fund_returns.csv", index_col=0, parse_dates=True)
    fams = ["equity", "crypto", "combined"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, fam in zip(axes, fams):
        cols = [c for c in r.columns if c.startswith(fam)]
        for c in cols:
            g = (1 + r[c].dropna()).cumprod()
            ax.plot(g.index, g.values, lw=1.6, label=c.replace(fam + "_", ""))
        ax.set_title(f"{fam.capitalize()} funds"); ax.set_ylabel("growth of $1")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Out-of-sample growth of $1 by fund family", fontweight="bold")
    fig.savefig(FIGS / "fig_equity_curves.png"); plt.close(fig)


def fig_risk_return():
    m = pd.read_csv(TABLES / "performance_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in m.iterrows():
        fam = _fam(row["fund"])
        ax.scatter(row["ann_vol"], row["ann_return"], s=90, color=FAM_COLOR[fam],
                   edgecolor="white", zorder=3)
        ax.annotate(row["fund"].replace(fam + "_", ""), (row["ann_vol"], row["ann_return"]),
                    fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    for fam, c in FAM_COLOR.items():
        ax.scatter([], [], color=c, label=fam)
    ax.set_xlabel("annualised volatility"); ax.set_ylabel("annualised return")
    ax.set_title("Risk and return of the 12 funds (out-of-sample)")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig_risk_return.png"); plt.close(fig)


def fig_sharpe_bars():
    m = pd.read_csv(TABLES / "performance_metrics.csv").sort_values("sharpe")
    colors = [FAM_COLOR[_fam(f)] for f in m["fund"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(m["fund"], m["sharpe"], color=colors)
    ax.set_xlabel("Sharpe ratio"); ax.set_title("Sharpe ratio by fund (out-of-sample)")
    fig.savefig(FIGS / "fig_sharpe_bars.png"); plt.close(fig)


def fig_sector_fear_greed():
    fg = pd.read_csv(DATA / "sector_fear_greed_100.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in fg.columns:
        ax.plot(fg.index, fg[c].rolling(21, min_periods=1).mean(), lw=1.4, label=c)
    ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Fear & Greed (0-100, 21d smooth)")
    ax.set_title("Sector news sentiment - raw Fear & Greed index")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.savefig(FIGS / "fig_sector_fear_greed.png"); plt.close(fig)


def fig_sector_zscore():
    z = pd.read_csv(DATA / "sector_sentiment_index.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in z.columns:
        ax.plot(z.index, z[c], lw=1.2, label=c)
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_ylabel("standardised sentiment (z)")
    ax.set_title("Sector sentiment - standardised signal (expanding window, no look-ahead)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.savefig(FIGS / "fig_sector_zscore.png"); plt.close(fig)


def fig_overlay_sharpe():
    f = pd.read_csv(TABLES / "fusion_metrics.csv")
    x = np.arange(len(f)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, f["base_sharpe"], w, label="base", color=BASE_C)
    ax.bar(x + w / 2, f["tilt_sharpe"], w, label="+ sentiment tilt", color=TILT_C)
    ax.set_xticks(x); ax.set_xticklabels(f["fund"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe ratio"); ax.set_title("Sentiment overlay: Sharpe, base vs tilt")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig_overlay_sharpe.png"); plt.close(fig)


def fig_overlay_drawdown():
    f = pd.read_csv(TABLES / "fusion_metrics.csv")
    x = np.arange(len(f)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, f["base_max_dd"], w, label="base", color=BASE_C)
    ax.bar(x + w / 2, f["tilt_max_dd"], w, label="+ sentiment tilt", color=TILT_C)
    ax.set_xticks(x); ax.set_xticklabels(f["fund"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("max drawdown"); ax.set_title("Sentiment overlay: max drawdown, base vs tilt")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig_overlay_drawdown.png"); plt.close(fig)


def fig_overlay_equity_curve(fund="combined_max_sharpe"):
    fr = pd.read_csv(DATA / "fusion_returns.csv", index_col=0, parse_dates=True)
    bcol, tcol = f"{fund}__base", f"{fund}__tilt"
    if bcol not in fr.columns:
        fund = fr.columns[0].split("__")[0]; bcol, tcol = f"{fund}__base", f"{fund}__tilt"
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((1 + fr[bcol].dropna()).cumprod(), color=BASE_C, lw=1.8, label="base")
    ax.plot((1 + fr[tcol].dropna()).cumprod(), color=TILT_C, lw=1.8, label="+ sentiment tilt")
    ax.set_ylabel("growth of $1"); ax.set_title(f"Sentiment overlay on {fund}")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "fig_overlay_equity_curve.png"); plt.close(fig)


def fig_sentiment_coverage():
    """Re-score headlines with base VADER vs finance-tuned to show the innovation."""
    from src import etl, sentiment
    news = etl.load_clean_news()
    base = sentiment.SentimentIntensityAnalyzer()
    fin = sentiment.build_finance_analyzer()
    cb = 100 * (sentiment.score_headlines(news, analyzer=base)["compound"] != 0).mean()
    cf = 100 * (sentiment.score_headlines(news, analyzer=fin)["compound"] != 0).mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(["base VADER", "finance-tuned\n(nVADER-lite)"], [cb, cf],
                  color=["#94a3b8", "#2563eb"])
    for b, v in zip(bars, [cb, cf]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("headlines with non-zero sentiment (%)")
    ax.set_title(f"Finance lexicon lifts coverage\n(+{fin.n_finance_terms_added} terms)")
    ax.set_ylim(0, 105)
    fig.savefig(FIGS / "fig_sentiment_coverage.png"); plt.close(fig)


def main():
    fig_equity_curves(); print("  fig_equity_curves")
    fig_risk_return(); print("  fig_risk_return")
    fig_sharpe_bars(); print("  fig_sharpe_bars")
    fig_sector_fear_greed(); print("  fig_sector_fear_greed")
    fig_sector_zscore(); print("  fig_sector_zscore")
    fig_overlay_sharpe(); print("  fig_overlay_sharpe")
    fig_overlay_drawdown(); print("  fig_overlay_drawdown")
    fig_overlay_equity_curve(); print("  fig_overlay_equity_curve")
    fig_sentiment_coverage(); print("  fig_sentiment_coverage")
    print("\nSaved 9 figures -> results/figures/")


if __name__ == "__main__":
    main()