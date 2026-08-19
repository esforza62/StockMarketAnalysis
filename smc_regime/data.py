"""OHLCV data fetching."""
from __future__ import annotations

import pandas as pd
import requests

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a ticker from Yahoo Finance's chart API."""
    response = requests.get(
        _CHART_URL.format(ticker=ticker),
        params={"range": period, "interval": interval},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()
    chart = response.json()["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")

    result = chart["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(result["timestamp"], unit="s"),
    ).dropna()

    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    return df
