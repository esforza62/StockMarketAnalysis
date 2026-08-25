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


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
) -> list[Trade]:
    """Simulate a single-position long-only strategy from entry/exit signals.

    stop_loss_pct, if set, closes the position at entry_price * (1 -
    stop_loss_pct/100) the first bar whose Low touches that level --
    checked ahead of that same bar's own exit signal, since a stop is a
    risk-management floor, not a strategy read on the bar's close.

    stop_loss_pct_series is the same idea but per-entry rather than one
    fixed percentage for every trade -- e.g. an ATR-based stop, where a
    volatile ticker gets a wider stop than a calm one. Looked up at the
    entry bar's date to fix that trade's own stop percentage for its
    whole duration (not re-computed every bar). Takes precedence over
    stop_loss_pct if both are given.

    max_hold_bars, if set, force-closes the position at that bar's Close
    once it has been held this many bars without hitting its own exit
    signal or stop -- caps how long a trade can sit waiting for an exit
    condition that may never come (rsi_dip_recovery's overbought exit, in
    particular, has no guarantee of ever firing if a dip just keeps
    falling).

    Neither a stop nor the time limit is ever checked on the entry bar
    itself: in_position only becomes True after that iteration's checks
    already ran, so nothing can close a position before it exists.
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    stop_price = None
    bars_held = 0

    for date, row in signals.iterrows():
        close = df.loc[date, "Close"]

        if in_position:
            bars_held += 1
            if stop_price is not None:
                low = df.loc[date, "Low"]
                if low <= stop_price:
                    trades.append(Trade(entry_date, date, entry_price, stop_price))
                    in_position = False
                    continue
            if max_hold_bars is not None and bars_held >= max_hold_bars:
                trades.append(Trade(entry_date, date, entry_price, close))
                in_position = False
                continue

        if not in_position and row["entry"]:
            in_position = True
            entry_date = date
            entry_price = close
            bars_held = 0
            if stop_loss_pct_series is not None:
                pct = stop_loss_pct_series.get(date)
                stop_price = entry_price * (1 - pct / 100) if pd.notna(pct) else None
            elif stop_loss_pct is not None:
                stop_price = entry_price * (1 - stop_loss_pct / 100)
            else:
                stop_price = None
        elif in_position and row["exit"]:
            trades.append(Trade(entry_date, date, entry_price, close))
            in_position = False

    return trades


def backtest_strategy(
    df: pd.DataFrame,
    strategy: str,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
) -> list[Trade]:
    signals = STRATEGIES[strategy](df)
    return run_backtest(
        df, signals,
        stop_loss_pct=stop_loss_pct, stop_loss_pct_series=stop_loss_pct_series, max_hold_bars=max_hold_bars,
    )
