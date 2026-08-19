"""OHLCV data fetching."""
from __future__ import annotations

import os
import re
from datetime import date

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta

TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"

_PERIOD_UNITS = {"d": "days", "mo": "months", "y": "years"}
_ADJUSTED_COLS = {
    "adjOpen": "Open",
    "adjHigh": "High",
    "adjLow": "Low",
    "adjClose": "Close",
    "adjVolume": "Volume",
}


def _period_to_start_date(period: str) -> str:
    """Convert a period string like '6mo', '1y', '5d' to an ISO start date."""
    match = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if not match:
        raise ValueError(f"Unsupported period {period!r}; use formats like '5d', '6mo', '2y'")
    count, unit = match.groups()
    return (date.today() - relativedelta(**{_PERIOD_UNITS[unit]: int(count)})).isoformat()


def _fetch_tiingo(ticker: str, period: str, interval: str, api_key: str) -> pd.DataFrame:
    if interval != "1d":
        raise ValueError(f"Tiingo source only supports interval='1d' (got {interval!r})")

    resp = requests.get(
        f"{TIINGO_BASE_URL}/{ticker}/prices",
        params={"startDate": _period_to_start_date(period), "format": "json", "token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["date"])
    df = df.rename(columns=_ADJUSTED_COLS)
    return df[list(_ADJUSTED_COLS.values())]


def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a ticker.

    Uses Tiingo when TIINGO_API_KEY is set, falling back to yfinance otherwise.
    """
    api_key = os.environ.get("TIINGO_API_KEY")
    if api_key:
        return _fetch_tiingo(ticker, period, interval, api_key)
    return _fetch_yfinance(ticker, period, interval)
