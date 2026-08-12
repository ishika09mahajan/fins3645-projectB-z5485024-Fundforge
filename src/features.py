"""Station 2 - your features: return features and headline text assembly.

Build your return (and optional risk) features here, and assemble the headlines
into a daily text panel. Part A is assembly only - scoring the text into sentiment
(and lagging the signal) is the Station 3 model, built in Part B, not here.
"""
import re
import pandas as pd
from collections import Counter


# ---------- return features ----------
def daily_returns_wide(prices, price_col="adjClose"):
    wide = prices.pivot(index="date", columns="ticker", values=price_col)
    return wide.pct_change(fill_method=None)


def combined_returns_panel(eq_prices, cr_prices):
    """Returns within each panel, then LEFT-MERGE crypto onto the equity calendar."""
    eq_ret = daily_returns_wide(eq_prices)
    cr_ret = daily_returns_wide(cr_prices)
    return eq_ret.join(cr_ret, how="left")


def descriptive_stats(returns_wide, label, ann_factor):
    r = pd.Series(returns_wide.to_numpy().ravel()).dropna()
    mean_d, std_d = r.mean(), r.std()
    return {
        "asset_class": label, "n_obs": int(r.shape[0]),
        "mean_daily_%": round(mean_d * 100, 3), "std_daily_%": round(std_d * 100, 3),
        "ann_return_%": round(mean_d * ann_factor * 100, 3),
        "ann_vol_%": round(std_d * (ann_factor ** 0.5) * 100, 3),
        "min_%": round(r.min() * 100, 3), "max_%": round(r.max() * 100, 3),
        "skew": round(r.skew(), 3), "excess_kurtosis": round(r.kurtosis(), 3),
    }


# ---------- text assembly (Part A = assembly + counting only, NO scoring) ----------
def assemble_headline_panel(headlines, trading_days):
    """Map each headline to its equity trading day: same day if trading, else next."""
    td = pd.DataFrame({"trading_day": pd.DatetimeIndex(sorted(trading_days))})
    td["trading_day"] = td["trading_day"].astype("datetime64[ns]")     # unify resolution
    h = headlines.sort_values("date").reset_index(drop=True)
    h["date"] = h["date"].astype("datetime64[ns]")                     # unify resolution
    mapped = pd.merge_asof(h[["date"]], td, left_on="date",
                           right_on="trading_day", direction="forward")
    h["trading_day"] = mapped["trading_day"].values
    return h.dropna(subset=["trading_day"]).reset_index(drop=True)


_WORD_RE = re.compile(r"[a-z']+")
_STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "is", "as", "at", "by",
         "with", "s", "its", "it", "this", "that", "from", "are", "be", "new", "will",
         "up", "down", "after", "amp", "you", "your"}
POS_WORDS = {"beat", "beats", "surge", "surges", "gain", "gains", "rise", "rises", "jump",
             "jumps", "record", "strong", "growth", "profit", "profits", "upgrade",
             "outperform", "rally", "boost", "soar", "soars", "win", "wins", "top", "tops",
             "high", "bullish"}
NEG_WORDS = {"miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges", "loss",
             "losses", "weak", "cut", "cuts", "downgrade", "decline", "declines", "slump",
             "fear", "fears", "warn", "warns", "lawsuit", "probe", "crash", "low", "bearish",
             "sink", "sinks"}


def _tokens(text):
    return _WORD_RE.findall(str(text).lower())


def word_frequencies(headlines, top_n=25):
    """Top words across headline titles (for the text-exploration figure)."""
    c = Counter()
    for title in headlines["title"]:
        c.update(w for w in _tokens(title) if w not in _STOP and len(w) > 1)
    return pd.DataFrame(c.most_common(top_n), columns=["word", "count"])


def sentiment_word_counts(headlines):
    """Count tone-bearing WORDS (descriptive vocabulary count, NOT a sentiment score)."""
    pos = neg = total = hits = 0
    for title in headlines["title"]:
        toks = _tokens(title)
        total += len(toks)
        p = sum(t in POS_WORDS for t in toks)
        n = sum(t in NEG_WORDS for t in toks)
        pos += p; neg += n
        if p + n > 0:
            hits += 1
    n = len(headlines)
    return {
        "n_headlines": n, "total_tokens": total,
        "avg_tokens_per_headline": round(total / n, 3),
        "positive_word_hits": pos, "negative_word_hits": neg,
        "headlines_with_tone_word": hits,
        "pct_headlines_with_tone_word": round(100 * hits / n, 3),
    }


def headlines_per_day(panel):
    """Article count per trading day (for the 'articles over time' figure)."""
    return panel.groupby("trading_day").size().rename("n_headlines").reset_index()