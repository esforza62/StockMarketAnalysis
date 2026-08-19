"""Regime-conditioned backtest harness.

Backtests each candidate strategy on each ticker, then tags every individual
trade with the SMC regime (and direction) that was active on its *entry*
date -- not the ticker's current regime. Aggregating those tags answers
"which strategy performs best in which regime?" directly from trade-level
outcomes, rather than proxying regime with a whole-period snapshot.
"""
from __future__ import annotations

import pandas as pd

from .backtest import backtest_strategy
from .data import fetch_ohlcv
from .regime import RegimeThresholds, classify_regime
from .strategies import STRATEGIES


def collect_trades(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    t: RegimeThresholds = RegimeThresholds(),
    start_date: str | None = None,
) -> pd.DataFrame:
    """Backtest every strategy on every ticker and tag each trade with the
    regime/direction active on its entry date."""
    records = []

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period, interval=interval, start_date=start_date)
        except Exception:
            continue
        regime = classify_regime(df, t)

        for strategy in STRATEGIES:
            for trade in backtest_strategy(df, strategy):
                if trade.entry_date not in regime.index:
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "regime": regime.loc[trade.entry_date, "regime"],
                        "direction": regime.loc[trade.entry_date, "direction"],
                        "entry_date": trade.entry_date,
                        "exit_date": trade.exit_date,
                        "return_pct": trade.return_pct,
                        "win": trade.return_pct > 0,
                    }
                )

    return pd.DataFrame.from_records(records)


def _ticker_compounded_return_pct(trades: pd.DataFrame) -> float:
    """Sequential compounding of one ticker's own trades, ordered by entry date.

    Valid here because this backtest only ever holds one position per ticker
    at a time -- a single ticker's trades genuinely do happen one after
    another for that ticker's own capital.
    """
    returns = trades.sort_values("entry_date")["return_pct"]
    growth = (1 + returns / 100).prod()
    return (growth - 1) * 100


def _compounded_return_pct(group: pd.DataFrame) -> float:
    """Equal-weighted average of each ticker's own compounded return.

    Compounding trades from DIFFERENT tickers together as if they were one
    sequential stream (the previous approach) is wrong: those trades happen
    concurrently in real time, not one after another, and chaining thousands
    of them as if serial explodes exponentially into meaningless numbers.
    Compounding within a ticker (realistic) and then averaging equally
    across tickers approximates splitting capital evenly across every
    ticker trading this strategy in this regime -- still a simplification,
    but not a nonsensical one.
    """
    per_ticker = group.groupby("ticker").apply(_ticker_compounded_return_pct)
    return per_ticker.mean()


def summarize_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level results into (regime, direction, strategy) performance.

    Trending and parabolic each split into "up" and "down" -- a long-only
    strategy set behaves very differently riding a trend up vs. trying to
    catch bounces in one going down, so collapsing direction away would
    hide exactly the asymmetry that matters. Choppy has no direction
    ("n/a") since it isn't trending either way.
    """

    def _agg(group: pd.DataFrame) -> pd.Series:
        returns = group["return_pct"]
        return pd.Series(
            {
                "trade_count": len(group),
                "win_rate": (returns > 0).mean() * 100,
                "avg_return_pct": returns.mean(),
                "total_return_pct": returns.sum(),
                "compounded_return_pct": _compounded_return_pct(group),
            }
        )

    summary = trades.groupby(["regime", "direction", "strategy"]).apply(_agg)
    summary["trade_count"] = summary["trade_count"].astype(int)
    return summary.reset_index().sort_values(["regime", "direction", "avg_return_pct"], ascending=[True, True, False])
