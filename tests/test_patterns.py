"""Bar-structure pattern detection in smc_regime.patterns, on hand-built bars.

Every case is a synthetic OHLC sequence whose right answer is decided by
construction, so a threshold change that quietly widens or narrows a
pattern shows up as a failing assert rather than as a different number in a
backtest nobody re-reads.
"""
import pandas as pd

from smc_regime import patterns as pat
from smc_regime.strategies import sweep_outside_reversal

# Doji-flat filler: Close == Open, so it is neither red nor green and each
# bar's high/low sit inside the previous one's -- no pattern can fire in the
# prefix, and it gives the ATR a calm baseline to warm up against.
FLAT = (100.0, 100.5, 99.5, 100.0, 1_000)


def make_df(bars, prefix=25):
    rows = [FLAT] * prefix + [(*bar, 1_000) for bar in bars]
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC"),
    )


def last(signals, column):
    return bool(signals[column].iloc[-1])


# --- bullish sweep-and-reclaim -------------------------------------------
# c1 red down to 98.5 (high 100.2); c2 red with a 3.0 lower wick sweeping
# c1's low at 98; c3 closes 100.8, back above c1's high.
C1 = (100.0, 100.2, 98.0, 98.5)
C2 = (98.4, 98.6, 95.0, 98.0)
C3 = (98.2, 101.0, 98.0, 100.8)

assert last(pat.sweep_reclaim(make_df([C1, C2, C3])), "bullish")

# The reclaim is the pattern. A third bar that rallies but closes at 100.0,
# just under c1's 100.2 high, is the same shape without the event.
assert not last(pat.sweep_reclaim(make_df([C1, C2, (98.2, 101.0, 98.0, 100.0)])), "bullish")

# c2 without the wick: a 0.1 tail under a 0.4 body is just a small red bar.
assert not last(pat.sweep_reclaim(make_df([C1, (98.4, 98.6, 97.9, 98.0), C3])), "bullish")

# c2 with the wick but no sweep of c1's low -- rejection of a level nobody's
# stops were under. Off by default, and the only difference require_sweep makes.
NO_SWEEP = (99.5, 99.6, 98.2, 99.4)
assert not last(pat.sweep_reclaim(make_df([C1, NO_SWEEP, C3])), "bullish")
assert last(pat.sweep_reclaim(make_df([C1, NO_SWEEP, C3]), require_sweep=False), "bullish")

# A green middle bar is a different pattern -- both of the first two candles
# have to be red for this one.
assert not last(pat.sweep_reclaim(make_df([C1, (95.5, 98.6, 95.0, 98.0), C3])), "bullish")

# --- bearish sweep-and-reclaim (the mirror) ------------------------------
BEAR = [(100.0, 102.0, 99.8, 101.5), (101.6, 105.0, 101.4, 102.0), (101.8, 102.0, 99.0, 99.5)]
bear_signals = pat.sweep_reclaim(make_df(BEAR))
assert last(bear_signals, "bearish")
assert not last(bear_signals, "bullish")
# Closing at 99.9, above c1's 99.8 low, is the failed version.
assert not last(pat.sweep_reclaim(make_df(BEAR[:2] + [(101.8, 102.0, 99.0, 99.9)])), "bearish")

# --- outside bars ---------------------------------------------------------
PREV = (100.0, 101.0, 99.0, 100.5)
assert last(pat.outside_bar(make_df([PREV, (99.5, 102.0, 98.5, 101.5)])), "bullish")
assert last(pat.outside_bar(make_df([PREV, (100.5, 102.0, 98.5, 98.6)])), "bearish")

# Higher high but not a lower low -- the range does not engulf, so it is not
# an outside bar however strong the close.
assert not last(pat.outside_bar(make_df([PREV, (99.5, 102.0, 99.5, 101.5)])), "bullish")

# Engulfing range, close back inside it: no direction under the default
# close-beyond rule, bullish under the looser colour-only one.
INSIDE_CLOSE = make_df([PREV, (99.5, 102.0, 98.5, 100.2)])
assert not last(pat.outside_bar(INSIDE_CLOSE), "bullish")
assert last(pat.outside_bar(INSIDE_CLOSE, require_close_beyond=False), "bullish")

# --- the strategy fires on either pattern --------------------------------
assert last(sweep_outside_reversal(make_df([C1, C2, C3])), "entry")
assert last(sweep_outside_reversal(make_df([PREV, (99.5, 102.0, 98.5, 101.5)])), "entry")
assert last(sweep_outside_reversal(make_df(BEAR)), "exit")
assert last(sweep_outside_reversal(make_df([PREV, (100.5, 102.0, 98.5, 98.6)])), "exit")

# Signals are plain booleans with no gaps -- run_backtest indexes them
# directly and a NaN there would be silently truthy.
signals = sweep_outside_reversal(make_df([C1, C2, C3]))
assert signals["entry"].dtype == bool and signals["exit"].dtype == bool
assert not signals.isna().any().any()

# --- no look-ahead --------------------------------------------------------
# The same bar must classify identically whether or not later bars exist:
# the whole point of an entry signal is that it was available at that close.
full = make_df([C1, C2, C3, (101.0, 103.0, 100.0, 102.5), (102.0, 104.0, 101.0, 103.5)])
truncated = full.iloc[: -2]
assert (
    pat.sweep_reclaim(full).loc[truncated.index[-1], "bullish"]
    == pat.sweep_reclaim(truncated).iloc[-1]["bullish"]
    == True
)

print("all pattern tests passed")
