"""Command-line entry point: classify a ticker's current regime, then look up
which strategy has historically performed best in that exact
(regime, direction) bucket from the accumulated daily-snapshot log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_ohlcv
from .regime import classify_regime


def load_summary(log_file: str, interval: str) -> list[dict]:
    records = [json.loads(line) for line in Path(log_file).read_text().splitlines() if line.strip()]
    matches = [r for r in records if r["interval"] == interval]
    if not matches:
        raise ValueError(f"No logged snapshots for interval={interval!r} in {log_file}")
    return matches[-1]["summary"]


def best_strategy(summary: list[dict], regime: str, direction: str, min_trades: int) -> dict | None:
    candidates = [
        row for row in summary
        if row["regime"] == regime and row["direction"] == direction and row["trade_count"] >= min_trades
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["avg_return_pct"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a strategy for a ticker's current regime.")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="6mo", help="lookback period for live regime classification")
    parser.add_argument("--interval", default="1d", help="bar interval, e.g. 1d, 1h")
    parser.add_argument("--log-file", default="backtest_logs/regime_strategy_log.jsonl", help="path to the snapshot log")
    parser.add_argument("--min-trades", type=int, default=20, help="minimum historical trades to trust a strategy match")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    latest = classify_regime(df).iloc[-1]
    regime, direction = latest["regime"], latest["direction"]

    print(f"{args.ticker.upper()}: currently '{regime}' ({direction})")

    summary = load_summary(args.log_file, args.interval)
    best = best_strategy(summary, regime, direction, args.min_trades)

    if best is None:
        print(f"No strategy has >= {args.min_trades} historical trades in this exact regime/direction bucket yet.")
        return

    print(
        f"  -> best fit: {best['strategy']} "
        f"(avg {best['avg_return_pct']:.2f}%/trade, {best['win_rate']:.0f}% win rate, "
        f"{best['trade_count']} historical trades, compounded {best['compounded_return_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
