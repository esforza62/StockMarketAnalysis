"""Daily snapshot job: run the regime-conditioned backtest across a tracking
universe at multiple intervals, and append a structured record to a log file.

Meant to run on a schedule (see the repo's Routine / cron setup). Each run
appends one JSON object per interval to the log, so 30 days of daily runs
build up a time series that can be checked for consistency -- does the same
strategy keep winning each regime, or does the ranking drift? -- and later
feed a dashboard.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .regime_backtest import collect_trades, summarize_by_regime

INTERVAL_PERIODS = {"1d": "2y", "1h": "730d"}


def run_snapshot(tickers: list[str], interval: str) -> dict:
    period = INTERVAL_PERIODS[interval]
    trades = collect_trades(tickers, period=period, interval=interval)

    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "interval": interval,
        "period": period,
        "ticker_count": len(tickers),
        "tickers_with_trades": int(trades["ticker"].nunique()) if not trades.empty else 0,
        "total_trades": int(len(trades)),
        "summary": [],
    }

    if not trades.empty:
        summary = summarize_by_regime(trades)
        record["summary"] = summary.round(4).to_dict(orient="records")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a daily regime-backtest snapshot and log the results.")
    parser.add_argument("--tickers-file", default="smc_regime/tracking_universe.txt", help="path to a newline-delimited ticker list")
    parser.add_argument("--intervals", default="1d,1h", help="comma-separated intervals to run, e.g. 1d,1h")
    parser.add_argument("--log-file", default="backtest_logs/regime_strategy_log.jsonl", help="JSONL file to append results to")
    args = parser.parse_args()

    tickers = [line.strip() for line in Path(args.tickers_file).read_text().splitlines() if line.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a") as f:
        for interval in intervals:
            record = run_snapshot(tickers, interval)
            f.write(json.dumps(record) + "\n")
            print(f"{interval}: {record['total_trades']} trades across {record['tickers_with_trades']}/{record['ticker_count']} tickers")


if __name__ == "__main__":
    main()
