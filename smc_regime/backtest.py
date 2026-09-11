"""Event-driven backtest: turns a strategy's entry/exit signals into a trade log.

Long-only unless a strategy asks otherwise. A signal frame carrying
"short_entry"/"short_exit" columns switches run_backtest into a
single-position long/short simulation; a frame without them takes exactly
the path it always did, so every existing strategy's trade log is
unchanged bar for bar.

Slippage is off by default (slippage_pct=0.0) so every historical number in
this repo stays comparable; passing it makes each fill the worse one, on both
entry and exit and on every exit path.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategies import STRATEGIES

LONG, SHORT = "long", "short"
_SHORT_COLUMNS = ("short_entry", "short_exit")


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str = LONG

    @property
    def return_pct(self) -> float:
        """Signed return on the position, so a short that fell in price wins.

        Called `side` and not `direction` on purpose: downstream --
        regime_backtest's records, the trades table -- "direction" already
        means the REGIME's direction, up or down. That is a property of the
        market; this is a property of the position taken in it, and the two
        are independent (a short in a trending-up regime is a normal thing
        to measure).
        """
        raw = (self.exit_price / self.entry_price - 1) * 100
        return -raw if self.side == SHORT else raw


def _stop_price(entry_price: float, stop_loss_pct: float, side: str) -> float:
    """A stop sits below entry for a long and above it for a short -- always
    on the losing side of the position."""
    sign = -1 if side == LONG else 1
    return entry_price * (1 + sign * stop_loss_pct / 100)


def _has_short_side(signals: pd.DataFrame) -> bool:
    present = [col for col in _SHORT_COLUMNS if col in signals.columns]
    if len(present) == 1:
        missing = next(col for col in _SHORT_COLUMNS if col not in present)
        raise ValueError(
            f"signals define {present[0]!r} without {missing!r} -- a strategy that can open "
            "shorts must also say what closes them"
        )
    return len(present) == 2


def _slipped(price: float, side: str, opening: bool, slippage_pct: float) -> float:
    """The fill you actually get, which is always the worse one.

    Opening a long or closing a short means BUYING, so you pay up; closing a
    long or opening a short means SELLING, so you receive less. One round
    trip therefore costs 2 x slippage_pct regardless of side, which is the
    point: a strategy holding 40 bars wears it once, a strategy holding 4
    bars wears it ten times as often per unit of time.

    Applied inside open/close rather than at the signal, so it covers every
    exit path -- signal, stop, max-hold and reversal -- identically. A stop
    is a level you wanted, not a fill you got.
    """
    if not slippage_pct:
        return price
    fraction = slippage_pct / 100
    buying = (opening and side == LONG) or (not opening and side == SHORT)
    return price * (1 + fraction) if buying else price * (1 - fraction)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
    slippage_pct: float = 0.0,
) -> list[Trade]:
    """Simulate a single-position strategy from entry/exit signals.

    stop_loss_pct, if set, closes the position stop_loss_pct away from
    entry on the losing side, the first bar whose Low (long) or High
    (short) touches that level -- checked ahead of that same bar's own exit
    signal, since a stop is a risk-management floor, not a strategy read on
    the bar's close.

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

    LONG/SHORT MODE, when signals carry "short_entry"/"short_exit", differs
    in exactly one further way: a position may close and the opposite one
    open on the SAME bar, at that bar's close. Without same-bar reversal a
    stop-and-reverse strategy -- where the bearish signal is both the
    long's exit and the short's entry -- could never take the short at all:
    it would go flat on the signal bar and the signal would be gone by the
    next one. Long-only mode keeps entering only from flat, so no existing
    strategy's results move.

    A bar firing the long AND short entry at once is ambiguous: any open
    position is closed and no new one is taken, rather than picking a side
    by column order.
    """
    long_short = _has_short_side(signals)
    trades: list[Trade] = []
    in_position = False
    side = LONG
    entry_date = entry_price = stop_price = None
    bars_held = 0

    def open_position(date, price: float, new_side: str) -> None:
        nonlocal in_position, side, entry_date, entry_price, stop_price, bars_held
        price = _slipped(price, new_side, True, slippage_pct)
        in_position, side, entry_date, entry_price, bars_held = True, new_side, date, price, 0
        pct = stop_loss_pct
        if stop_loss_pct_series is not None:
            looked_up = stop_loss_pct_series.get(date)
            pct = looked_up if pd.notna(looked_up) else None
        stop_price = _stop_price(price, pct, new_side) if pct is not None else None

    def close_position(date, price: float) -> None:
        nonlocal in_position
        trades.append(Trade(entry_date, date, entry_price, _slipped(price, side, False, slippage_pct), side))
        in_position = False

    for date, row in signals.iterrows():
        close = df.loc[date, "Close"]

        if in_position:
            bars_held += 1
            if stop_price is not None:
                touched = (
                    df.loc[date, "Low"] <= stop_price if side == LONG
                    else df.loc[date, "High"] >= stop_price
                )
                if touched:
                    close_position(date, stop_price)
                    continue
            if max_hold_bars is not None and bars_held >= max_hold_bars:
                close_position(date, close)
                continue

        if not long_short:
            if not in_position and row["entry"]:
                open_position(date, close, LONG)
            elif in_position and row["exit"]:
                close_position(date, close)
            continue

        go_long, go_short = bool(row["entry"]), bool(row["short_entry"])
        if in_position:
            closes_out = row["exit"] if side == LONG else row["short_exit"]
            reverses = go_short if side == LONG else go_long
            if closes_out or reverses:
                close_position(date, close)
        if not in_position and go_long != go_short:
            open_position(date, close, LONG if go_long else SHORT)

    return trades


def backtest_strategy(
    df: pd.DataFrame,
    strategy: str,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
    slippage_pct: float = 0.0,
) -> list[Trade]:
    signals = STRATEGIES[strategy](df)
    return run_backtest(
        df, signals,
        stop_loss_pct=stop_loss_pct, stop_loss_pct_series=stop_loss_pct_series,
        max_hold_bars=max_hold_bars, slippage_pct=slippage_pct,
    )
