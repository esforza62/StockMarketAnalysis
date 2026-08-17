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
) -> pd.DataFrame:
    """Backtest every strategy on every ticker and tag each trade with the
    regime/direction active on its entry date."""
    records = []

    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period=period, interval=interval)
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


def _compounded_return_pct(group: pd.DataFrame) -> float:
    """Sequential compounding of every trade's return, ordered by entry date.

    Simplified vs. a real portfolio: assumes one trade at a time (capital
    fully reinvested each trade, no concurrent positions), so it overstates
    what a real multi-ticker account would do -- but it's a much closer
    approximation of "what would this have made me" than a raw sum of
    percentages, which ignores compounding entirely.
    """
    returns = group.sort_values("entry_date")["return_pct"]
    growth = (1 + returns / 100).prod()
    return (growth - 1) * 100


def summarize_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level results into (regime, strategy) performance."""

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

    summary = trades.groupby(["regime", "strategy"]).apply(_agg)
    summary["trade_count"] = summary["trade_count"].astype(int)
    return summary.reset_index().sort_values(["regime", "avg_return_pct"], ascending=[True, False])
