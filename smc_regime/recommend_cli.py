"""Command-line entry point: classify a ticker's current regime, then look up
which strategy has historically performed best in that exact
(regime, direction) bucket, falling back from symbol-specific stats to
sector-level stats to the full-population pooled stats when a ticker
doesn't have enough of its own trade history yet."""
from __future__ import annotations

import argparse

from . import db as db_module
from .data import fetch_ohlcv
from .metadata import get_industry, get_sector
from .regime import classify_regime


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a strategy for a ticker's current regime.")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="6mo", help="lookback period for live regime classification")
    parser.add_argument("--interval", default="1d", help="bar interval, e.g. 1d, 1h")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH), help="path to the SQLite snapshot store")
    parser.add_argument("--min-trades", type=int, default=15, help="minimum historical trades to trust a tier's strategy match")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    latest = classify_regime(df).iloc[-1]
    regime, direction = latest["regime"], latest["direction"]

    print(f"{args.ticker.upper()}: currently '{regime}' ({direction})")

    conn = db_module.connect(args.db_file)
    sector = get_sector(args.ticker)
    industry = get_industry(args.ticker)
    best = db_module.best_strategy(
        conn, args.interval, regime, direction,
        ticker=args.ticker, industry=industry, sector=sector, min_trades=args.min_trades,
    )
    conn.close()

    if best is None:
        print(f"No strategy has >= {args.min_trades} historical trades in this exact regime/direction bucket yet.")
        return

    tier_label = {
        "symbol": f"{args.ticker.upper()}-specific",
        "industry": f"{industry} industry group",
        "sector": f"{sector} sector",
        "pooled": "full population",
    }[best["source"]]
    print(
        f"  -> best fit: {best['strategy']} "
        f"(avg {best['avg_return_pct']:.2f}%/trade, {best['win_rate']:.0f}% win rate, "
        f"{best['trade_count']} trades, basis: {tier_label})"
    )


if __name__ == "__main__":
    main()
