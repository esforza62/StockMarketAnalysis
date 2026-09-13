"""Per-trade matched control: did the entry beat entering at random?

summarize_by_regime() reports what a strategy earned. It has no control to
subtract, so a strategy that holds for months in a bull market posts a
large average return per trade whether or not its entry rule contributed
anything. That is not a hypothetical -- it is what the vfi_zero_cross
numbers turned out to be (see its docstring in strategies.py).

The control here is matched per trade rather than per bucket, because
holding periods are heavily right-skewed: a strategy averaging 85 calendar
days can have a median hold of 12 bars, and comparing its mean return
against a median-horizon baseline compares two different things.

For a trade on ticker T held H bars, entered while T was in regime R, the
control is the mean H-bar forward return from *every* bar of T that was
also in regime R. That is what a coin-flip entry on the same name, in the
same regime, over the same horizon would have earned. Edge is the mean of
(trade return - its own control).

Limits, stated rather than buried: trades overlap in time, so the paired
t-statistic's independence assumption is optimistic -- read |t| as a screen
for "not obviously noise", not as a p-value. Everything is gross of costs,
which flatters high-turnover strategies most. And the control is drawn from
the same history the strategy traded, so it measures entry timing within a
regime, not whether trading that regime was a good idea at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Trade


def _bucket_labels(regime_df: pd.DataFrame) -> np.ndarray:
    return (regime_df["regime"] + "/" + regime_df["direction"]).to_numpy()


def control_returns(df: pd.DataFrame, regime_df: pd.DataFrame, trades: list[Trade]) -> pd.DataFrame:
    """One row per trade: its return, its matched control, and the difference.

    Trades whose entry or exit falls outside `regime_df`, and those whose
    horizon leaves no comparable starting bar (a long hold in a regime the
    ticker was rarely in), are dropped rather than compared against a
    partial control.
    """
    close = df["Close"].to_numpy()
    position = {date: i for i, date in enumerate(df.index)}
    buckets = _bucket_labels(regime_df)
    cache: dict[tuple[str, int], float] = {}

    def control(bucket: str, horizon: int) -> float:
        key = (bucket, horizon)
        if key not in cache:
            starts = np.flatnonzero((buckets == bucket) & (np.arange(len(close)) + horizon < len(close)))
            cache[key] = (
                np.nan if starts.size == 0
                else float(((close[starts + horizon] / close[starts] - 1) * 100).mean())
            )
        return cache[key]

    records = []
    for trade in trades:
        if trade.entry_date not in position or trade.exit_date not in position:
            continue
        horizon = position[trade.exit_date] - position[trade.entry_date]
        if horizon <= 0:
            continue
        bucket = buckets[position[trade.entry_date]]
        expected = control(bucket, horizon)
        if np.isnan(expected):
            continue
        records.append(
            {
                "bucket": bucket,
                "hold_bars": horizon,
                "return_pct": trade.return_pct,
                "control_pct": expected,
                "edge_pct": trade.return_pct - expected,
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["bucket", "hold_bars", "return_pct", "control_pct", "edge_pct"]
    )


def summarize_edge(control_df: pd.DataFrame, min_trades: int = 30) -> pd.DataFrame:
    """Aggregate control_returns() output into per-bucket edge, with a
    paired t-statistic on (trade - its own control)."""

    def _agg(group: pd.DataFrame) -> pd.Series:
        edge = group["edge_pct"]
        spread = edge.std(ddof=1) if len(edge) > 1 else np.nan
        return pd.Series(
            {
                "trade_count": len(group),
                "median_hold_bars": group["hold_bars"].median(),
                "strategy_return_pct": group["return_pct"].mean(),
                "control_return_pct": group["control_pct"].mean(),
                "edge_pct": edge.mean(),
                "t_stat": edge.mean() / (spread / np.sqrt(len(edge))) if spread else np.nan,
            }
        )

    if control_df.empty:
        return pd.DataFrame(
            columns=["bucket", "trade_count", "median_hold_bars", "strategy_return_pct",
                     "control_return_pct", "edge_pct", "t_stat"]
        )
    summary = control_df.groupby("bucket").apply(_agg, include_groups=False).reset_index()
    summary["trade_count"] = summary["trade_count"].astype(int)
    return summary[summary["trade_count"] >= min_trades].sort_values("edge_pct", ascending=False)
