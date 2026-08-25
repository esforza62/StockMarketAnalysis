"""Event-driven long-only backtest: turns a strategy's entry/exit signals into a trade log."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategies import STRATEGIES


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1) * 100


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, stop_loss_pct: float | None = None) -> list[Trade]:
    """Simulate a single-position long-only strategy from entry/exit signals.

    stop_loss_pct, if set, closes the position at entry_price * (1 -
    stop_loss_pct/100) the first bar whose Low touches that level --
    checked ahead of that same bar's own exit signal, since a stop is a
    risk-management floor, not a strategy read on the bar's close. Never
    checked on the entry bar itself: in_position only becomes True after
    the stop-check runs for that iteration, so a stop can't fire before
    the position exists.
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    stop_price = None

    for date, row in signals.iterrows():
        close = df.loc[date, "Close"]

        if in_position and stop_price is not None:
            low = df.loc[date, "Low"]
            if low <= stop_price:
                trades.append(Trade(entry_date, date, entry_price, stop_price))
                in_position = False
                continue

        if not in_position and row["entry"]:
            in_position = True
            entry_date = date
            entry_price = close
            stop_price = entry_price * (1 - stop_loss_pct / 100) if stop_loss_pct is not None else None
        elif in_position and row["exit"]:
            trades.append(Trade(entry_date, date, entry_price, close))
            in_position = False

    return trades


def backtest_strategy(df: pd.DataFrame, strategy: str, stop_loss_pct: float | None = None) -> list[Trade]:
    signals = STRATEGIES[strategy](df)
    return run_backtest(df, signals, stop_loss_pct=stop_loss_pct)
