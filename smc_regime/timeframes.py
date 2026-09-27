"""Session-anchored aggregation for the 15m -> 1h -> 4h -> 1d ladder.

Bar-shape patterns are interval-specific (see patterns.py), so reading one
across a ladder means building the higher rungs from the base series rather
than fetching each separately: aggregating UP is exact, and it guarantees
every rung describes the same tape. The reverse is impossible, which is why
the base interval has to be the finest one in the ladder.

WHAT THE RUNGS ACTUALLY ARE. A US equity RTH session is 390 minutes, so the
ladder does not nest evenly and the shortfall lands in the last bar of each
session:

    15m   26 bars/session   exact
    1h     7 bars/session   6 full bars + a 30-minute stub (15:30-16:00)
    4h     2 bars/session   09:30-13:30 full, then a 2.5-hour "4h" bar
    1d     1 bar            6.5h / 4h = 1.625, not a whole number of 4h bars

The ragged bars are kept deliberately: this is exactly what a charting
platform shows, and a signal the code sees but the chart does not is worse
than an uneven bar. Verified against Yahoo's own hourly bars -- session
resampling 15m to 1h reproduces them, 419 of 420 closes agreeing to the
cent over 60 days of AAPL.

Sessions are resampled one at a time, anchored to each session's own first
bar, so a bin never spans the overnight gap and the grid re-anchors after
a holiday or a late open instead of drifting.

WHY BAR SLOT MATTERS. Bars at different times of day are not comparable
even when their durations match. Measured over 60 days: AAPL's 09:30 4h bar
averages 2.14% range against 0.99% for the afternoon bar, and equalising
the durations (two 195-minute bars) barely moves it -- 2.10% vs 1.09%. That
is the intraday volatility smile, not the clock. Any ATR-relative threshold
therefore over-qualifies opening bars and under-qualifies midday ones
unless the ATR baseline is per-slot; see patterns.slot_atr.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import patterns as pat

MARKET_TZ = "America/New_York"

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

#: Rungs above a 15-minute base, as pandas offset aliases. "240min" is the
#: ragged 4h described above, not a true four-hour bar.
LADDER = ("60min", "240min", "1D")


def _session_dates(index: pd.DatetimeIndex):
    """Trading-session date for each bar. Intraday timestamps are converted
    to market time first: a 16:00 ET bar is the next UTC day, and grouping
    on the UTC date would split one session across two groups."""
    return (index.tz_convert(MARKET_TZ) if index.tz is not None else index).date


def _base_bar_length(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing, so the overnight gap doesn't count as a bar."""
    if len(index) < 2:
        return pd.Timedelta(0)
    return pd.Series(index).diff().median()


def _is_session_rule(rule: str) -> bool:
    """True for the daily rung, which is one bar per SESSION rather than a
    fixed-length bin.

    pandas only honours `origin` for tick-like frequencies (minutes,
    hours), and silently ignores it for "1D" -- which would anchor daily
    bins to midnight UTC and split a session in two. Anything coarser than
    a day is rejected outright rather than quietly mis-binned: weekly bars
    already have their own path in data._resample_weekly, built from daily.
    """
    offset = pd.tseries.frequencies.to_offset(rule)
    if isinstance(offset, pd.tseries.offsets.Tick) and offset.nanos < pd.Timedelta(days=1).value:
        return False
    if offset.n == 1 and offset.name in ("D", "B"):
        return True
    raise ValueError(
        f"rung {rule!r} is coarser than a session -- the ladder tops out at one bar per day"
    )


def session_resample(df: pd.DataFrame, rule: str, min_source_bars: int = 1) -> pd.DataFrame:
    """Aggregate to `rule`, one session at a time, anchored to the session open.

    Returns the OHLCV columns plus:
      bar_end     -- when this bar finished, i.e. the first moment its
                     values were knowable. Aggregating gives a bar the
                     timestamp it STARTED at, which is the wrong stamp to
                     align on: treating a 09:30 4h bar as known at 09:30
                     would let a signal read three and a half hours of its
                     own future. Everything downstream aligns on bar_end.
      source_bars -- how many base bars went into it, which is what makes
                     a ragged bar visible instead of silent.

    min_source_bars drops bins built from fewer than that many base bars.
    Only relevant for a rule that divides the session unevenly enough to
    leave a bin holding just the closing print (195min does; 240min does
    not, its remainder joins the afternoon bar). Left at 1 by default so
    nothing is dropped without being asked for.
    """
    if df.empty:
        return df.assign(bar_end=pd.NaT, source_bars=0).iloc[:0]

    bar_len = _base_bar_length(df.index)
    whole_session = _is_session_rule(rule)
    frames = []
    for _, session in df.groupby(_session_dates(df.index)):
        if whole_session:
            # A trading day is 6.5 hours, not 24, so the daily rung cannot be
            # a 24-hour bin -- it is the session itself, collapsed to one bar.
            agg = session.resample(rule).agg(_AGG).dropna(subset=["Open"])
            agg = agg.iloc[[0]] if len(agg) else agg
            agg["source_bars"] = len(session)
            agg["bar_end"] = session.index[-1] + bar_len
            frames.append(agg)
            continue
        binned = session.resample(rule, origin=session.index[0])
        agg = binned.agg(_AGG)
        agg["source_bars"] = binned.size()
        agg["bar_end"] = session.index.to_series().resample(rule, origin=session.index[0]).max() + bar_len
        frames.append(agg.dropna(subset=["Open"]))

    out = pd.concat(frames).sort_index()
    return out[out["source_bars"] >= min_source_bars]


def align_to_base(values: pd.Series, bar_end: pd.Series, base_index: pd.DatetimeIndex) -> pd.Series:
    """Carry a higher-timeframe series onto the base index, using only bars
    that had already closed.

    Forward-filled from bar_end, so a base bar sees the most recent
    COMPLETED higher-timeframe bar and never the one it is currently inside
    -- the whole point of aligning on bar_end rather than bar start. A base
    bar that lands before the first higher bar closes has no reading yet
    and gets False.
    """
    stamped = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(bar_end.to_numpy()))
    stamped = stamped[~stamped.index.duplicated(keep="last")].sort_index()
    aligned = stamped.reindex(base_index, method="ffill")
    if values.dtype == bool:
        return aligned.fillna(False).astype(bool)
    return aligned


def higher_timeframe_state(
    df: pd.DataFrame,
    rule: str,
    detector: Callable[[pd.DataFrame], pd.DataFrame] = pat.bar_direction,
    min_source_bars: int = 1,
) -> pd.DataFrame:
    """Run `detector` on the `rule` aggregation of df, mapped back onto df's
    own index as of the last higher bar to have closed."""
    htf = session_resample(df, rule, min_source_bars=min_source_bars)
    if htf.empty:
        false = pd.Series(False, index=df.index)
        return pd.DataFrame({"bullish": false, "bearish": false})

    signals = detector(htf)
    return pd.DataFrame(
        {
            col: align_to_base(signals[col], htf["bar_end"], df.index)
            for col in ("bullish", "bearish")
        }
    )


def ladder_state(
    df: pd.DataFrame,
    rules: tuple[str, ...] = LADDER,
    detector: Callable[[pd.DataFrame], pd.DataFrame] = pat.bar_direction,
) -> pd.DataFrame:
    """Every rung's most recent completed read, aligned to the base index.

    Columns are "<rule>_bullish"/"<rule>_bearish", plus "rungs_bullish" and
    "rungs_bearish" counting how many rungs agree -- which is the thing to
    correlate trade outcomes against, rather than assuming agreement helps.

    The default detector is patterns.bar_direction (which way the rung's
    last closed bar resolved), not patterns.reversal_signals. Asking
    whether the 4h ALSO printed the three-bar pattern answers "no" for
    almost every bar and collapses the split into a single bucket; asking
    which way the 4h is pointing is both denser and closer to what
    "higher timeframe agrees" is normally taken to mean. Pass
    detector=patterns.reversal_signals for the strict reading.
    """
    out = {}
    for rule in rules:
        state = higher_timeframe_state(df, rule, detector=detector)
        out[f"{rule}_bullish"] = state["bullish"]
        out[f"{rule}_bearish"] = state["bearish"]

    frame = pd.DataFrame(out, index=df.index)
    frame["rungs_bullish"] = frame[[f"{r}_bullish" for r in rules]].sum(axis=1)
    frame["rungs_bearish"] = frame[[f"{r}_bearish" for r in rules]].sum(axis=1)
    return frame
