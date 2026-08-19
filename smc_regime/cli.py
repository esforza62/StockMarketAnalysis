"""Command-line entry point: classify a ticker's recent price-movement regime."""
from __future__ import annotations

import argparse

from .data import fetch_ohlcv
from .regime import classify_regime

DISPLAY_COLS = [
    "efficiency_ratio",
    "adx",
    "atr_pct",
    "atr_pct_roc",
    "ema_extension_atr",
    "regime",
    "direction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a stock's recent price-movement regime.")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="6mo", help="lookback period, e.g. 3mo, 6mo, 1y, 2y")
    parser.add_argument("--interval", default="1d", help="bar interval, e.g. 1d, 1h")
    parser.add_argument("--tail", type=int, default=10, help="rows to print")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    result = classify_regime(df)

    print(result[DISPLAY_COLS].tail(args.tail).round(3).to_string())

    latest = result.iloc[-1]
    print(f"\n{args.ticker.upper()}: currently '{latest['regime']}' ({latest['direction']})")


if __name__ == "__main__":
    main()
