# Your project code. data_access.py is provided; the rest is yours.
"""Innovation - a custom finance-sentiment lexicon (ported from Part A).

Part A used this to COUNT tone words and measure coverage. In Part B the same
word lists extend VADER (see src/sentiment.py): the finance words VADER lacks
are injected so headline tone is read in a finance-aware way.
"""
import re
import pandas as pd

POSITIVE = {
    "beat", "beats", "surge", "surges", "surged", "soar", "soars", "soared", "jump",
    "jumps", "jumped", "rally", "rallies", "rallied", "gain", "gains", "gained", "rise",
    "rises", "rose", "climb", "climbs", "climbed", "rebound", "rebounds", "outperform",
    "outperforms", "record", "high", "highs", "top", "tops", "topped", "profit",
    "profits", "profitable", "growth", "grow", "grows", "grew", "strong", "stronger",
    "strength", "upgrade", "upgrades", "upgraded", "raise", "raised", "boost", "boosts",
    "boosted", "win", "wins", "winner", "winning", "bullish", "buy", "expansion",
    "dividend", "dividends", "buyback", "buybacks", "upside", "momentum", "optimistic",
    "optimism", "accelerate",
}
NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "slump", "slumps",
    "slumped", "sink", "sinks", "sank", "tumble", "tumbles", "tumbled", "fall", "falls",
    "fell", "drop", "drops", "dropped", "decline", "declines", "declined", "slide",
    "slides", "slid", "crash", "crashes", "crashed", "loss", "losses", "lose", "loses",
    "lost", "weak", "weaker", "weakness", "downgrade", "downgrades", "downgraded", "cut",
    "cuts", "cutting", "warn", "warns", "warned", "warning", "lawsuit", "lawsuits",
    "probe", "probes", "investigation", "fraud", "bankruptcy", "bankrupt", "default",
    "recession", "fear", "fears", "bearish", "sell", "selloff", "downside", "risk",
    "risks", "slowdown", "layoffs", "layoff", "underperform", "low", "lows", "concern",
    "concerns", "trouble", "struggle", "struggles",
}
LEXICON = POSITIVE | NEGATIVE

_WORD_RE = re.compile(r"[a-z']+")


def _tokens(text):
    return _WORD_RE.findall(str(text).lower())


def tag_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    df = headlines.copy()
    pos, neg, ntok = [], [], []
    for title in df["title"]:
        tks = _tokens(title)
        pos.append(sum(t in POSITIVE for t in tks))
        neg.append(sum(t in NEGATIVE for t in tks))
        ntok.append(len(tks))
    df["pos_hits"] = pos
    df["neg_hits"] = neg
    df["tone_hits"] = df["pos_hits"] + df["neg_hits"]
    df["n_tokens"] = ntok
    df["has_tone"] = df["tone_hits"] > 0
    return df


def coverage_by_sector(tagged: pd.DataFrame) -> pd.DataFrame:
    g = tagged.groupby("sector")
    out = g.agg(
        headlines=("title", "size"),
        pct_with_tone=("has_tone", lambda s: round(100 * s.mean(), 3)),
        tone_hits=("tone_hits", "sum"),
        tokens=("n_tokens", "sum"),
    ).reset_index()
    out["tone_per_1000_tokens"] = round(1000 * out["tone_hits"] / out["tokens"], 3)
    return out.sort_values("pct_with_tone", ascending=False).reset_index(drop=True)


def coverage_over_time(tagged: pd.DataFrame) -> pd.DataFrame:
    s = tagged.set_index("date")["has_tone"].resample("ME").mean() * 100
    return s.round(3).rename("pct_with_tone").reset_index()


def overall_coverage(tagged: pd.DataFrame) -> dict:
    return {
        "lexicon_size": len(LEXICON),
        "positive_words": len(POSITIVE),
        "negative_words": len(NEGATIVE),
        "headlines": len(tagged),
        "pct_with_tone": round(100 * tagged["has_tone"].mean(), 3),
    }