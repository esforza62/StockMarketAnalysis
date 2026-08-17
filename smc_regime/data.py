"""OHLCV data fetching."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a ticker via yfinance."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    return df[["Open", "High", "Low", "Close", "Volume"]]
