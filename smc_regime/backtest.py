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


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame) -> list[Trade]:
    """Simulate a single-position long-only strategy from entry/exit signals."""
    trades = []
    in_position = False
    entry_date = None
    entry_price = None

    for date, row in signals.iterrows():
        close = df.loc[date, "Close"]
        if not in_position and row["entry"]:
            in_position = True
            entry_date = date
            entry_price = close
        elif in_position and row["exit"]:
            trades.append(Trade(entry_date, date, entry_price, close))
            in_position = False

    return trades


def backtest_strategy(df: pd.DataFrame, strategy: str) -> list[Trade]:
    signals = STRATEGIES[strategy](df)
    return run_backtest(df, signals)
