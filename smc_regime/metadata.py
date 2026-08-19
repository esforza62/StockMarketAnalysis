"""Ticker metadata: exchange + GICS sector, used to support a fallback
inference tier (symbol-specific -> sector-level -> full-population pooled)
for tickers that don't yet have enough of their own trade history.

Sector comes from the scraped `smc_regime/sectors/*.txt` GICS files, which
only cover S&P 500 constituents. The tracking universe also includes
non-S&P names (foreign large caps, ETFs, recent IPOs) -- those are covered
by SECTOR_OVERRIDES below, hand-mapped to their GICS sector.

Exchange comes from Tiingo's `/tiingo/daily/{ticker}` metadata endpoint
(no `/prices` suffix), cached to disk since it almost never changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

_SECTORS_DIR = Path(__file__).parent / "sectors"
_CACHE_PATH = Path(__file__).parent / "data" / "exchange_cache.json"
_META_URL = "https://api.tiingo.com/tiingo/daily/{ticker}"

# ETFs and non-S&P-500 names not covered by the scraped GICS sector files,
# hand-mapped to their GICS sector.
SECTOR_OVERRIDES = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "AAL": "Industrials",
    "ALAB": "Information Technology",
    "ALNY": "Health Care",
    "ARM": "Information Technology",
    "ASML": "Information Technology",
    "CCEP": "Consumer Staples",
    "CRWV": "Information Technology",
    "FER": "Industrials",
    "MELI": "Consumer Discretionary",
    "MSTR": "Information Technology",
    "NBIS": "Information Technology",
    "NIO": "Consumer Discretionary",
    "PDD": "Consumer Discretionary",
    "QS": "Industrials",
    "RKLB": "Industrials",
    "SHOP": "Information Technology",
    "SPCX": "Industrials",
    "TRI": "Industrials",
}


def load_sector_map() -> dict[str, str]:
    sector_map: dict[str, str] = {}
    for path in _SECTORS_DIR.glob("*.txt"):
        sector = path.stem.replace("_", " ").title()
        for line in path.read_text().splitlines():
            ticker = line.strip()
            if ticker:
                sector_map[ticker] = sector
    return sector_map


def get_sector(ticker: str, sector_map: dict[str, str] | None = None) -> str:
    sector_map = sector_map if sector_map is not None else load_sector_map()
    ticker = ticker.upper()
    return sector_map.get(ticker) or SECTOR_OVERRIDES.get(ticker) or "Unknown"


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_exchange(ticker: str, cache: dict[str, str] | None = None) -> str:
    """Fetch a ticker's exchange code from Tiingo, cached to disk since it
    essentially never changes. Pass a shared `cache` dict when resolving
    many tickers in a loop to batch the disk write."""
    ticker = ticker.upper()
    owns_cache = cache is None
    if owns_cache:
        cache = _load_cache()

    if ticker not in cache:
        token = os.environ.get("TIINGO_API_KEY")
        if not token:
            raise RuntimeError("TIINGO_API_KEY environment variable is not set")
        response = requests.get(_META_URL.format(ticker=ticker), params={"token": token}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        cache[ticker] = payload.get("exchangeCode") or "Unknown"
        if owns_cache:
            _save_cache(cache)

    return cache[ticker]


def build_ticker_metadata(tickers: list[str]) -> list[dict[str, str]]:
    """Resolve {ticker, exchange, sector} for a list of tickers, one API
    call per not-yet-cached ticker."""
    sector_map = load_sector_map()
    cache = _load_cache()
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "ticker": ticker.upper(),
                "exchange": get_exchange(ticker, cache=cache),
                "sector": get_sector(ticker, sector_map=sector_map),
            }
        )
    _save_cache(cache)
    return rows
