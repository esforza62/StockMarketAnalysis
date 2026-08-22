"""Stock-split contamination guard for intraday (hourly/15m) trades.

Tiingo's daily EOD endpoint has adjusted-price fields (adjOpen/High/Low/
Close), so data.py uses those and daily/weekly are clean. Tiingo's IEX
intraday endpoint has no adjusted-price equivalent, so hourly/15m bars
still carry the raw nominal price -- meaning a trade whose entry/exit
window straddles a real stock split shows a fake near-total-wipeout
return (confirmed: NVDA's 2021-07-20 4:1 split alone produced a -72.8%
"loss" on an hourly rsi_dip_recovery trade; a universe-wide scan found 230
of 669,048 hourly trades contaminated this way, rare but severe -- several
at -90% to -98%).

Split dates are fetched from Yahoo's chart API (the same public endpoint
already used in cross_validate.py) and cached to a git-committed JSON file
rather than fetched live during the nightly pipeline: splits are rare
events, and there's no reason to make the core Tiingo-based pipeline
depend on a second, undocumented external data source at runtime just for
this. Refresh the cache periodically with the CLI below, not every night.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

_CACHE_PATH = Path(__file__).parent / "data" / "split_cache.json"
_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Intervals whose underlying data isn't split-adjusted -- daily/weekly are
# fixed at the source (data.py uses Tiingo's adjClose) and don't need this.
UNADJUSTED_INTERVALS = {"1h", "1hour", "15m", "15min", "5m", "5min", "1m", "1min", "30m", "30min"}


def fetch_split_dates(ticker: str, range_: str = "10y") -> list[pd.Timestamp]:
    resp = requests.get(
        _YAHOO_URL.format(ticker=ticker),
        params={"range": range_, "interval": "1d", "events": "splits"},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"]
    if not result:
        return []
    events = result[0].get("events", {}).get("splits") or {}
    return [pd.Timestamp(int(ts), unit="s", tz="UTC") for ts in events.keys()]


def load_split_cache(cache_path: Path | str = _CACHE_PATH) -> dict[str, list[pd.Timestamp]]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    return {ticker: [pd.Timestamp(d) for d in dates] for ticker, dates in raw.get("splits", {}).items()}


def save_split_cache(splits: dict[str, list[pd.Timestamp]], cache_path: Path | str = _CACHE_PATH) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
        "splits": {ticker: [d.isoformat() for d in dates] for ticker, dates in splits.items()},
    }
    cache_path.write_text(json.dumps(payload, indent=2))


def refresh_cache(tickers: list[str], cache_path: Path | str = _CACHE_PATH) -> dict[str, list[pd.Timestamp]]:
    """Fetch current split history for every ticker and overwrite the cache.
    Only tickers with at least one split are stored (keeps the file small)."""
    splits: dict[str, list[pd.Timestamp]] = {}
    for ticker in tickers:
        try:
            dates = fetch_split_dates(ticker)
        except Exception as exc:
            print(f"  split fetch failed for {ticker}: {exc}")
            continue
        if dates:
            splits[ticker] = dates
    save_split_cache(splits, cache_path)
    return splits


def filter_contaminated_trades(trades: pd.DataFrame, splits: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    """Drop rows whose entry_date/exit_date straddle a known split for that
    ticker. `trades` must have ticker/entry_date/exit_date columns; dates
    may be epoch seconds (int) or already tz-aware Timestamps -- matches
    either the in-memory collect_trades() output or a DB-read frame."""
    if trades.empty or not splits:
        return trades

    entry = trades["entry_date"]
    exit_ = trades["exit_date"]
    if not pd.api.types.is_datetime64_any_dtype(entry):
        entry = pd.to_datetime(entry, unit="s", utc=True)
        exit_ = pd.to_datetime(exit_, unit="s", utc=True)

    contaminated = pd.Series(False, index=trades.index)
    for ticker, dates in splits.items():
        ticker_mask = trades["ticker"] == ticker
        if not ticker_mask.any():
            continue
        for split_date in dates:
            contaminated |= ticker_mask & (entry < split_date) & (exit_ > split_date)

    return trades[~contaminated]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the stock-split date cache used to filter contaminated intraday trades.")
    parser.add_argument("--tickers-file", default="smc_regime/tracking_universe.txt")
    parser.add_argument("--cache-file", default=str(_CACHE_PATH))
    args = parser.parse_args()

    tickers = [line.strip() for line in Path(args.tickers_file).read_text().splitlines() if line.strip()]
    print(f"Fetching split history for {len(tickers)} tickers from Yahoo...")
    splits = refresh_cache(tickers, args.cache_file)
    total_splits = sum(len(d) for d in splits.values())
    print(f"{len(splits)} tickers have split history ({total_splits} total split events). Cache written to {args.cache_file}")


if __name__ == "__main__":
    main()
