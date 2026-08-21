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

# Tiingo's IEX intraday endpoint hard-caps responses at 10,000 rows and
# silently returns the most recent 10,000 rather than erroring or paging
# -- confirmed by testing directly (an explicit limit=50000 made no
# difference). A 7.6-year hourly request is ~11,000+ bars, so a single
# request for our full history silently drops everything before roughly
# the most recent 5.7 years with no error to notice. Chunking each
# intraday request to a window we've confirmed stays well under the cap
# (a 2-year/730-day window was ~3,650 bars in earlier testing) avoids
# this; daily requests never approach the cap (~1,900 bars over 7.6
# years) so they don't need it.
#
# Chunk size has to shrink with bar density: 700 days of 15-minute bars
# (~4x hourly's density) would land north of 14,000 bars and hit the same
# cap it's meant to dodge. Each entry keeps roughly the same ~3,650-bar
# safety margin that was empirically confirmed for hourly, scaled down by
# how much denser that interval is.
_INTRADAY_CHUNK_DAYS = {
    "1min": 12, "1m": 12,
    "5min": 60, "5m": 60,
    "15min": 175, "15m": 175,
    "30min": 350, "30m": 350,
    "1hour": 700, "1h": 700,
}
_DEFAULT_INTRADAY_CHUNK_DAYS = 175

_WEEKLY_INTERVALS = {"1w", "1wk", "1week", "w"}


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


def _fetch_chunk(ticker: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, token: str) -> pd.DataFrame:
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
        raise ValueError(f"No data returned for {ticker!r} (interval={interval!r})")
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        raise ValueError(f"Tiingo error for {ticker!r}: {payload.get('detail', payload)}")
    if not payload:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars from daily -- Tiingo's EOD endpoint has no native weekly
    mode, and weekly volume is low enough (~1 bar/week) that resampling
    already-fetched daily data locally is cheaper than a separate fetch
    path anyway. Weeks are labeled by their last trading day (Friday),
    not pandas' default week-ending-Sunday convention, to match how
    weekly bars are normally read on a chart."""
    weekly = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna()


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d", start_date: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker from Tiingo (EOD for daily, IEX for
    intraday, resampled from daily for weekly).

    `start_date` (e.g. "2019-01-01"), when given, overrides `period` with a
    fixed calendar anchor instead of a rolling lookback from today.
    """
    if interval.lower() in _WEEKLY_INTERVALS:
        daily = fetch_ohlcv(ticker, period=period, interval="1d", start_date=start_date)
        return _resample_weekly(daily)

    token = _api_key()
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = pd.Timestamp(start_date, tz="UTC") if start_date else _period_to_start(period, end)
    chunk_days = _INTRADAY_CHUNK_DAYS.get(interval, _DEFAULT_INTRADAY_CHUNK_DAYS)

    if interval == "1d" or (end - start).days <= chunk_days:
        df = _fetch_chunk(ticker, interval, start, end, token)
    else:
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
            chunks.append(_fetch_chunk(ticker, interval, chunk_start, chunk_end, token))
            chunk_start = chunk_end + pd.Timedelta(days=1)
        df = pd.concat(chunks) if chunks else pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = df[~df.index.duplicated(keep="first")].sort_index()

    if df.empty:
        raise ValueError(f"No data returned for {ticker!r} (period={period!r}, interval={interval!r})")
    return df
