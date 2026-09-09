"""Candlestick reversal patterns as vectorised boolean masks.

Each detector returns a DataFrame with "bullish" and "bearish" columns
aligned to the input index, so a strategy can OR several of them together
(see strategies.sweep_outside_reversal) without each one re-deriving bodies and
wicks.

Deliberately no trend/context filter inside the detectors, matching
rsi_dip_recovery's reasoning: every trade is tagged downstream with the SMC
regime active on entry, and "which regime does this pattern actually work
in" is precisely what the regime-conditioned backtest exists to measure.
Baking a trend gate in here would answer that question by construction
instead of measuring it.

INTERVAL SENSITIVITY, measured rather than assumed (Yahoo 15m/1h, 60 days,
AAPL/NVDA/SPY): these are bar-shape patterns, so the bar grid is part of the
signal, not a neutral container for it. Roughly 40 bullish signals fire on a
15m series where ~10 fire on the 1h series covering the same tape, and only
about 5 of them are the same event -- most 15m patterns are invisible at 1h,
and about half of the 1h patterns have no 15m counterpart. Worse for
intuition: re-anchoring the SAME data to a 1h grid offset by 15, 30 or 45
minutes keeps only ~40-65% of the signals. Some fraction of any single
interval's pattern count is an artifact of where the session boundary
happens to fall.

Two consequences. The interval is part of the strategy definition, so a
result on 1d says nothing about 15m and each interval has to be measured on
its own (which the nightly snapshot already does -- it runs 1d/1h/1w/15m
separately). And a strategy wanting genuine multi-timeframe confirmation
should aggregate UP from the finest series it has, since 15m bars can be
resampled into exact 1h bars while the reverse is impossible -- see
strategies.vwap_multi_timeframe on why a true cross-interval signal needs
backtest.py's one-interval-at-a-time contract extended first.

Thresholds are expressed as multiplications, never ratios: a doji's body is
legitimately 0.0, and `wick >= mult * body` handles that (trivially true,
still gated by the range fraction and the ATR floor) where `wick / body`
would divide by zero.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind


def _parts(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Body size and the two wick lengths, colour-agnostic."""
    open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    return {
        "body": (close - open_).abs(),
        "range": high - low,
        "upper_wick": high - body_top,
        "lower_wick": body_bottom - low,
    }


def sweep_reclaim(
    df: pd.DataFrame,
    wick_body_mult: float = 2.0,
    wick_range_frac: float = 0.5,
    require_sweep: bool = True,
    atr_window: int = 14,
    min_range_atr: float = 0.5,
) -> pd.DataFrame:
    """Three-bar sweep-and-reclaim reversal.

    Bullish: two red candles, the second carrying a long lower wick (and,
    with require_sweep, printing a low BELOW the first candle's low -- the
    stop run), then a third bar closing back above the FIRST candle's high.

    The reclaim is close-based, not a wick touch: requiring the close past
    the level is what separates "buyers took the level back" from "price
    tagged it and failed," and it is the whole point of the pattern -- both
    red candles' sellers are underwater at that close, and their covering
    is the move being traded. A touch-based variant fires far more often
    and is a different, looser measurement; if it's wanted it belongs as
    its own parameter with its own backtest, not as a silent relaxation of
    this one.

    require_sweep is the difference between "the second bar had a tail" and
    "liquidity below the prior low was taken and rejected." On by default
    because the untaken-low version is both much more common and much
    weaker; turn it off to measure that difference rather than assume it.

    min_range_atr keeps the wick tests from qualifying a bar so small that
    its "long wick" is a couple of ticks of noise: the second candle's own
    high-low range must be at least this many ATRs. Bars before the ATR
    warmup completes are NaN and therefore never qualify.

    Bearish is the exact mirror: two green candles, the second with a long
    upper wick sweeping above the first's high, then a close below the
    first candle's low.
    """
    open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
    parts = _parts(df)
    atr = ind.atr(df, atr_window)

    # c1 is two bars back, c2 one bar back, c3 the bar being evaluated --
    # the signal is only known at c3's close, so nothing here reads ahead.
    red = close < open_
    green = close > open_
    big_enough = (parts["range"] >= atr * min_range_atr).shift(1).fillna(False)

    long_lower = (
        (parts["lower_wick"] >= parts["body"] * wick_body_mult)
        & (parts["lower_wick"] >= parts["range"] * wick_range_frac)
    ).shift(1).fillna(False)
    long_upper = (
        (parts["upper_wick"] >= parts["body"] * wick_body_mult)
        & (parts["upper_wick"] >= parts["range"] * wick_range_frac)
    ).shift(1).fillna(False)

    swept_low = (low.shift(1) < low.shift(2)) if require_sweep else True
    swept_high = (high.shift(1) > high.shift(2)) if require_sweep else True

    bullish = (
        red.shift(2).fillna(False)
        & red.shift(1).fillna(False)
        & long_lower
        & swept_low
        & big_enough
        & (close > high.shift(2))
    )
    bearish = (
        green.shift(2).fillna(False)
        & green.shift(1).fillna(False)
        & long_upper
        & swept_high
        & big_enough
        & (close < low.shift(2))
    )
    return pd.DataFrame(
        {"bullish": bullish.fillna(False), "bearish": bearish.fillna(False)},
        index=df.index,
    )


def outside_bar(df: pd.DataFrame, require_close_beyond: bool = True) -> pd.DataFrame:
    """Two-bar outside (engulfing-range) reversal.

    The bar's range covers the prior bar's entirely -- higher high AND
    lower low -- so both sides of the previous bar's action were traded
    through in one bar.

    require_close_beyond decides how the direction is read. On (default):
    the close must finish beyond the prior bar's opposite extreme, the same
    "reclaim the level" standard sweep_reclaim uses, which keeps the two
    detectors measuring the same kind of event when they're OR'd together.
    Off: direction falls back to the bar's own colour, which is the looser
    textbook outside-bar definition and fires several times more often.
    """
    open_, high, low, close = df["Open"], df["High"], df["Low"], df["Close"]
    engulfs = (high > high.shift(1)) & (low < low.shift(1))

    if require_close_beyond:
        bullish = engulfs & (close > high.shift(1))
        bearish = engulfs & (close < low.shift(1))
    else:
        bullish = engulfs & (close > open_)
        bearish = engulfs & (close < open_)

    return pd.DataFrame(
        {"bullish": bullish.fillna(False), "bearish": bearish.fillna(False)},
        index=df.index,
    )
