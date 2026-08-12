"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
from __future__ import annotations

import pandas as pd
from src import data_access

END_DATE = "2023-12-31"


def load_clean_equities() -> pd.DataFrame:
    df = data_access.load_equity_prices()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END_DATE].copy()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_clean_crypto() -> pd.DataFrame:
    df = data_access.load_crypto_prices()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END_DATE].copy()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_clean_news() -> pd.DataFrame:
    df = data_access.load_news_headlines()
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df[df["date"] <= END_DATE].copy()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def count_price_duplicates(prices: pd.DataFrame) -> int:
    return int(prices.duplicated(subset=["ticker", "date"]).sum())


def count_news_duplicates(news: pd.DataFrame) -> int:
    return int(news.duplicated(subset=["ticker", "date", "title"]).sum())


def dedupe_news(news: pd.DataFrame) -> pd.DataFrame:
    return news.drop_duplicates(subset=["ticker", "date", "title"]).reset_index(drop=True)