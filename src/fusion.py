"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolios

TILT_STRENGTH = 0.5   # fixed ex ante: a +1 sd sector gets ~1.65x its base weight


def ticker_sector_map(equities: pd.DataFrame) -> dict:
    """ticker -> sector, from the equities panel."""
    return (equities[["ticker", "sector"]].drop_duplicates()
            .set_index("ticker")["sector"].to_dict())


def sector_signal_to_ticker(sector_signal: pd.DataFrame, tk_sector: dict) -> pd.DataFrame:
    cols = {tk: sector_signal[sec] for tk, sec in tk_sector.items()
            if sec in sector_signal.columns}
    return pd.DataFrame(cols, index=sector_signal.index)


def apply_sentiment(weights: pd.DataFrame, sentiment: pd.DataFrame,
                    strength: float = TILT_STRENGTH) -> pd.DataFrame:
    sig = sentiment.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    tilted = weights * np.exp(strength * sig)
    return tilted.div(tilted.sum(axis=1), axis=0)


def compare_base_vs_tilt(returns: pd.DataFrame, tilt_ticker: pd.DataFrame, method: str,
                         periods_per_year: int = 252, strength: float = TILT_STRENGTH,
                         **kw) -> dict:
    base = portfolios.oos_backtest(returns, method=method, cov_method="lw",
                                   periods_per_year=periods_per_year, **kw)
    tilted = portfolios.oos_backtest(returns, method=method, cov_method="lw",
                                     periods_per_year=periods_per_year,
                                     tilt=tilt_ticker, tilt_strength=strength, **kw)
    return {"base": base, "tilt": tilted}