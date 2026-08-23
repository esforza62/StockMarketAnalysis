"""Command-line entry point: backtest candidate strategies across tickers,
conditioned on the SMC regime active when each trade was entered."""
from __future__ import annotations

import argparse

from .regime_backtest import collect_trades, summarize_by_regime


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest strategies by regime across tickers.")
    parser.add_argument("tickers", nargs="+", help="Tickers to backtest, e.g. AAPL MSFT TSLA")
    parser.add_argument("--period", default="2y", help="lookback period, e.g. 1y, 2y")
    parser.add_argument("--interval", default="1d", help="bar interval, e.g. 1d")
    parser.add_argument("--min-trades", type=int, default=5, help="hide (regime, strategy) rows with fewer trades than this")
    args = parser.parse_args()

    trades, _ = collect_trades(args.tickers, period=args.period, interval=args.interval)
    if trades.empty:
        print("No trades generated.")
        return

    summary = summarize_by_regime(trades)
    shown = summary[summary["trade_count"] >= args.min_trades]
    print(shown.round(2).to_string(index=False))

    print(f"\n{len(trades)} total trades across {trades['ticker'].nunique()} tickers.")
    print("\nBest strategy per regime+direction (by avg return/trade, min-trades filter applied):")
    for (regime, direction), group in shown.groupby(["regime", "direction"]):
        if group.empty:
            continue
        best = group.iloc[0]
        label = regime if direction == "n/a" else f"{regime} ({direction})"
        print(
            f"  {label:<18} -> {best['strategy']} "
            f"(avg {best['avg_return_pct']:.2f}%/trade, {int(best['trade_count'])} trades, {best['win_rate']:.0f}% win rate)"
        )


if __name__ == "__main__":
    main()
