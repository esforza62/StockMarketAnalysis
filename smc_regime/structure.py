"""Market structure: swing points, BOS/CHoCH, and fair value gaps.

patterns.py answers "what shape is this bar". This module answers the two
questions a shape alone cannot: has the structure changed, and is price
somewhere that matters. A rejection bar is tradeable because it happens at
an untested imbalance inside an already-broken structure -- three different
kinds of thing, and only the first is a candlestick.

    swing points   a local extreme, confirmed `length` bars after the fact
    BOS            close beyond the last swing WITH the trend -- continuation
    CHoCH          the first close beyond it AGAINST the trend -- the reversal
    FVG            a three-bar imbalance, tracked until price fills it

NO LOOK-AHEAD, and it is easy to get wrong here. A pivot high needs
`length` bars on BOTH sides to be a pivot at all, so it is not knowable on
the bar it occurred -- only `length` bars later. Everything this module
returns is stamped at the bar where the information became available, not
the bar it describes: a swing high found at bar i is reported at bar
i+length, carrying the original bar's price and timestamp. Reporting it at
bar i would let a strategy trade a level it could not have drawn yet, which
inflates every result and shows up nowhere in the output.

FVGs need no such shift: the three bars forming one are all in the past by
the time the third closes.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BULLISH, BEARISH = 1, -1


def swing_points(df: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """Confirmed pivot highs and lows.

    A pivot high is a bar whose high is the highest of the `length` bars
    either side of it. Ties count as pivots -- a flat top is still a level
    price respected, and requiring a strict maximum silently drops the
    double-top case that matters most.

    Columns are stamped at the CONFIRMING bar (`length` bars after the
    pivot), and carry the pivot's own price and timestamp so a caller can
    still draw the level in the right place:

        high_confirmed / low_confirmed  bool
        high_price / low_price          the pivot bar's high / low
        high_time / low_time            the pivot bar's timestamp
    """
    window = 2 * length + 1
    high, low = df["High"], df["Low"]

    is_pivot_high = high == high.rolling(window, center=True).max()
    is_pivot_low = low == low.rolling(window, center=True).min()

    stamps = pd.Series(df.index, index=df.index)
    return pd.DataFrame(
        {
            "high_confirmed": is_pivot_high.shift(length).fillna(False).astype(bool),
            "low_confirmed": is_pivot_low.shift(length).fillna(False).astype(bool),
            "high_price": high.shift(length),
            "low_price": low.shift(length),
            "high_time": stamps.shift(length),
            "low_time": stamps.shift(length),
        },
        index=df.index,
    )


def structure_events(df: pd.DataFrame, length: int = 5) -> pd.DataFrame:
    """BOS and CHoCH from the confirmed swing sequence.

    The trend is whichever direction last broke a level. A close beyond the
    most recent swing high is bullish; whether it is a BOS or a CHoCH
    depends only on what the trend was a moment before -- the same break is
    continuation in an uptrend and a reversal in a downtrend. That is the
    whole definition, and it is why CHoCH cannot be computed from price
    shape alone: it needs the previous state.

    A level is consumed when it breaks, so one swing high cannot produce a
    BOS on every subsequent bar. The next break needs a newly confirmed
    swing to aim at.

    Columns: bos_bullish, bos_bearish, choch_bullish, choch_bearish, and
    `trend` (+1/-1/0 before the first break).
    """
    swings = swing_points(df, length)
    close = df["Close"]

    records = []
    trend = 0
    pending_high = pending_low = None

    for date in df.index:
        row = swings.loc[date]
        if row["high_confirmed"]:
            pending_high = row["high_price"]
        if row["low_confirmed"]:
            pending_low = row["low_price"]

        bos_up = bos_down = choch_up = choch_down = False
        price = close.loc[date]

        if pending_high is not None and price > pending_high:
            if trend == BEARISH:
                choch_up = True
            else:
                bos_up = True
            trend = BULLISH
            pending_high = None
        elif pending_low is not None and price < pending_low:
            if trend == BULLISH:
                choch_down = True
            else:
                bos_down = True
            trend = BEARISH
            pending_low = None

        records.append(
            {
                "bos_bullish": bos_up, "bos_bearish": bos_down,
                "choch_bullish": choch_up, "choch_bearish": choch_down,
                "trend": trend,
            }
        )

    return pd.DataFrame(records, index=df.index)


@dataclass
class Gap:
    direction: int
    top: float
    bottom: float
    created: pd.Timestamp


def fair_value_gaps(
    df: pd.DataFrame,
    min_size_atr: float = 0.0,
    atr_window: int = 14,
    max_age_bars: int | None = None,
) -> pd.DataFrame:
    """Three-bar imbalances, tracked from creation until price fills them.

    A bullish gap exists when a bar's low sits above the high of the bar
    two back: price moved so fast that the range in between never traded.
    Bearish is the mirror. Both are known as soon as the third bar closes,
    so unlike swings there is nothing to shift.

    A gap is "touched" on any bar whose range overlaps it, and removed once
    price closes the whole thing -- trading through the far edge, not
    merely dipping in. That distinction is the point of the pattern: the
    first tap is the setup, and a gap that has been fully traded through is
    no longer an imbalance at all.

    min_size_atr ignores gaps thinner than a fraction of ATR. Left at 0 by
    default: on any liquid intraday series most bars leave a tick-wide gap,
    and whether filtering them helps is a question to measure, not to
    assume in the detector.

    Columns: fvg_bullish/fvg_bearish (a gap was CREATED here),
    in_bullish_fvg/in_bearish_fvg (price is inside an unfilled one now),
    and open_bullish/open_bearish counts.
    """
    high, low = df["High"], df["Low"]
    from . import indicators as ind

    threshold = ind.atr(df, atr_window) * min_size_atr if min_size_atr else None

    active: list[Gap] = []
    records = []
    prev_high2 = high.shift(2)
    prev_low2 = low.shift(2)

    for i, date in enumerate(df.index):
        bar_high, bar_low = high.loc[date], low.loc[date]

        # Price interacts with gaps that already existed before this bar.
        touched_bull = any(g.direction == BULLISH and bar_low <= g.top and bar_high >= g.bottom for g in active)
        touched_bear = any(g.direction == BEARISH and bar_low <= g.top and bar_high >= g.bottom for g in active)
        active = [
            g for g in active
            if not (
                (g.direction == BULLISH and bar_low < g.bottom)
                or (g.direction == BEARISH and bar_high > g.top)
            )
        ]
        if max_age_bars is not None:
            active = [g for g in active if (df.index.get_loc(date) - df.index.get_loc(g.created)) <= max_age_bars]

        created_bull = created_bear = False
        if i >= 2:
            h2, l2 = prev_high2.loc[date], prev_low2.loc[date]
            floor = threshold.loc[date] if threshold is not None else 0.0
            if bar_low > h2 and (bar_low - h2) >= floor:
                active.append(Gap(BULLISH, bar_low, h2, date))
                created_bull = True
            elif bar_high < l2 and (l2 - bar_high) >= floor:
                active.append(Gap(BEARISH, l2, bar_high, date))
                created_bear = True

        records.append(
            {
                "fvg_bullish": created_bull, "fvg_bearish": created_bear,
                "in_bullish_fvg": touched_bull, "in_bearish_fvg": touched_bear,
                "open_bullish": sum(1 for g in active if g.direction == BULLISH),
                "open_bearish": sum(1 for g in active if g.direction == BEARISH),
            }
        )

    return pd.DataFrame(records, index=df.index)
