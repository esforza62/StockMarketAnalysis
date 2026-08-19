"""OHLCV data fetching."""
from __future__ import annotations

import os
import re

import pandas as pd
import requests

_DAILY_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
_IEX_URL = "https://api.tiingo.com/iex/{ticker}/prices"

_INTRADAY_FREQ = {
    "1m": "1min", "1min": "1min",
    "5m": "5min", "5min": "5min",
    "15m": "15min", "15min": "15min",
    "30m": "30min", "30min": "30min",
    "1h": "1hour", "1hour": "1hour",
}


def _api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise RuntimeError("TIINGO_API_KEY environment variable is not set")
    return key


def _period_to_start(period: str, end: pd.Timestamp) -> pd.Timestamp:
    match = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if not match:
        raise ValueError(f"Unsupported period format: {period!r} (expected e.g. '6mo', '2y', '730d')")
    n, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        return end - pd.Timedelta(days=n)
    if unit == "mo":
        return end - pd.DateOffset(months=n)
    return end - pd.DateOffset(years=n)


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d", start_date: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker from Tiingo (EOD for daily, IEX for intraday).

    `start_date` (e.g. "2019-01-01"), when given, overrides `period` with a
    fixed calendar anchor instead of a rolling lookback from today.
    """
    token = _api_key()
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = pd.Timestamp(start_date, tz="UTC") if start_date else _period_to_start(period, end)
    date_params = {"startDate": start.strftime("%Y-%m-%d"), "endDate": end.strftime("%Y-%m-%d")}

    if interval == "1d":
        url = _DAILY_URL.format(ticker=ticker)
        params = {"token": token, "format": "json", **date_params}
    else:
        freq = _INTRADAY_FREQ.get(interval)
        if freq is None:
            raise ValueError(f"Unsupported interval: {interval!r}")
        url = _IEX_URL.format(ticker=ticker)
        params = {
            "token": token,
            "format": "json",
            "resampleFreq": freq,
            "columns": "open,high,low,close,volume",
            **date_params,
        }

    response = requests.get(url, params=params, timeout=20)
    if response.status_code == 404:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        raise ValueError(f"Tiingo error for {ticker!r}: {payload.get('detail', payload)}")
    if not payload:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")

    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    return df
