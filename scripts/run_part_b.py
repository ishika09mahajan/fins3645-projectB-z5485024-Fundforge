"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from src import etl, features, portfolios, sentiment, fusion  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"
DATA.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def build_funds():
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()
    eq_ret = features.daily_returns_wide(eq)
    cr_ret = features.daily_returns_wide(cr)
    combined = features.combined_returns_panel(eq, cr)
    families = {"equity": (eq_ret, 252), "crypto": (cr_ret, 365), "combined": (combined, 252)}
    fund_returns, fund_weights, rows = {}, {}, []
    for fam, (R, ppy) in families.items():
        for m in portfolios.METHODS:
            out = portfolios.oos_backtest(R, method=m, window=252, cov_method="lw",
                                          cost_bps=10.0, periods_per_year=ppy)
            name = f"{fam}_{m}"
            fund_returns[name] = out["daily_net"]
            fund_weights[name] = out["weights"].iloc[-1]
            rows.append({"fund": name, **out["metrics"]})
            print(f"  built {name}")
    perf = pd.DataFrame(rows)
    eq[["ticker", "sector"]].drop_duplicates().to_csv(DATA / "ticker_sector.csv", index=False)
    pd.DataFrame(fund_returns).to_csv(DATA / "fund_returns.csv")
    pd.DataFrame(fund_weights).to_csv(DATA / "fund_weights.csv")
    perf.to_csv(TABLES / "performance_metrics.csv", index=False)
    return perf


def build_sentiment():
    news = etl.load_clean_news()
    sia = sentiment.build_finance_analyzer()
    scored = sentiment.score_headlines(news, analyzer=sia)
    idx = sentiment.build_sector_index(scored)
    idx["smoothed"].to_csv(DATA / "sector_sentiment_index.csv")
    idx["raw_100"].to_csv(DATA / "sector_fear_greed_100.csv")
    cov = 100.0 * (scored["compound"] != 0).mean()
    print(f"  scored {len(scored)} headlines | finance terms added "
          f"{sia.n_finance_terms_added} | non-zero coverage {cov:.1f}%")
    print(f"  sectors: {list(idx['smoothed'].columns)} | days {idx['smoothed'].shape[0]}")
    return scored, idx


def build_fusion(sector_signal):
    """Sentiment overlay: tilt each equity/combined fund by its sector's lagged tone."""
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()
    eq_ret = features.daily_returns_wide(eq)
    combined = features.combined_returns_panel(eq, cr)
    tk_sector = fusion.ticker_sector_map(eq)
    tilt = fusion.sector_signal_to_ticker(sector_signal, tk_sector)

    families = {"equity": (eq_ret, 252), "combined": (combined, 252)}
    rows, fusion_returns = [], {}
    for fam, (R, ppy) in families.items():
        for m in portfolios.METHODS:
            res = fusion.compare_base_vs_tilt(R, tilt, m, periods_per_year=ppy)
            b, t = res["base"]["metrics"], res["tilt"]["metrics"]
            name = f"{fam}_{m}"
            fusion_returns[f"{name}__base"] = res["base"]["daily_net"]
            fusion_returns[f"{name}__tilt"] = res["tilt"]["daily_net"]
            rows.append({
                "fund": name,
                "base_ann_return": b["ann_return"], "tilt_ann_return": t["ann_return"],
                "base_sharpe": b["sharpe"], "tilt_sharpe": t["sharpe"],
                "base_max_dd": b["max_drawdown"], "tilt_max_dd": t["max_drawdown"],
                "delta_sharpe": round(t["sharpe"] - b["sharpe"], 4),
            })
            print(f"  fused {name}: sharpe {b['sharpe']:.3f} -> {t['sharpe']:.3f}")
    fus = pd.DataFrame(rows)
    pd.DataFrame(fusion_returns).to_csv(DATA / "fusion_returns.csv")
    fus.to_csv(TABLES / "fusion_metrics.csv", index=False)
    return fus


def main():
    print("Building 12 funds (walk-forward OOS - takes a few minutes)...")
    perf = build_funds()
    print("\n=== FUND PERFORMANCE (out-of-sample) ===")
    print(perf.to_string(index=False))

    print("\nScoring headlines and building the sector sentiment index...")
    _, idx = build_sentiment()

    print("\nFusing sentiment into the equity/combined funds (base vs tilt)...")
    fus = build_fusion(idx["smoothed"])
    print("\n=== SENTIMENT OVERLAY (base vs tilt) ===")
    print(fus.to_string(index=False))

    print("\nSaved -> results/data/{fund_returns,fund_weights,sector_sentiment_index,"
          "sector_fear_greed_100,fusion_returns}.csv")
    print("      -> results/tables/{performance_metrics,fusion_metrics}.csv")


if __name__ == "__main__":
    main()