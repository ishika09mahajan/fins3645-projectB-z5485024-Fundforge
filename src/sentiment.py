"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src import lexicon

POS_VALENCE = 1.5     # VADER valence scale is about -4..+4
NEG_VALENCE = -1.5


def build_finance_analyzer(pos_valence: float = POS_VALENCE,
                           neg_valence: float = NEG_VALENCE) -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    added = 0
    for w in lexicon.POSITIVE:
        if w not in sia.lexicon:
            sia.lexicon[w] = pos_valence
            added += 1
    for w in lexicon.NEGATIVE:
        if w not in sia.lexicon:
            sia.lexicon[w] = neg_valence
            added += 1
    sia.n_finance_terms_added = added
    return sia


def score_headlines(headlines: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None,
                    text_col: str = "title") -> pd.DataFrame:
    sia = analyzer if analyzer is not None else build_finance_analyzer()
    df = headlines.copy()
    df["compound"] = [sia.polarity_scores(str(t))["compound"] for t in df[text_col]]
    return df


def to_fear_greed_100(compound):
    return (compound + 1.0) / 2.0 * 100.0


def ticker_day_sentiment(scored: pd.DataFrame, day_col: str = "date") -> pd.DataFrame:
    g = scored.groupby([day_col, "ticker"])["compound"].mean().reset_index()
    return g.pivot(index=day_col, columns="ticker", values="compound").sort_index()


def sector_sentiment_index(scored: pd.DataFrame, day_col: str = "date") -> pd.DataFrame:
    td = scored.groupby([day_col, "sector", "ticker"])["compound"].mean().reset_index()
    sec = td.groupby([day_col, "sector"])["compound"].mean().reset_index()
    wide = sec.pivot(index=day_col, columns="sector", values="compound").sort_index()
    return to_fear_greed_100(wide)


def expanding_zscore(df: pd.DataFrame, min_periods: int = 30) -> pd.DataFrame:
    mean = df.expanding(min_periods=min_periods).mean()
    sd = df.expanding(min_periods=min_periods).std()
    return (df - mean) / sd


def build_sector_index(scored: pd.DataFrame, smooth_window: int = 21,
                       day_col: str = "date") -> dict:
    raw = sector_sentiment_index(scored, day_col)
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    raw = raw.reindex(full).ffill()
    z = expanding_zscore(raw)
    smoothed = z.rolling(smooth_window, min_periods=1).mean().shift(1)
    return {"raw_100": raw, "zscore": z, "smoothed": smoothed}