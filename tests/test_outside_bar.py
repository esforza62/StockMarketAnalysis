"""Bar-anatomy checks for the outside-bar port in smc_regime.strategies.

Synthetic bars throughout -- these run without a Tiingo key. Each case is a
hand-built three-bar sequence where only one condition of the Pine original
is in question, so a wrong answer points straight at the condition that
broke rather than at "the signal count changed".
"""
import pandas as pd

from smc_regime.strategies import _outside_bar_signals, outside_bar_reversal, outside_bar_rsi


def frame(bars):
    """bars: list of (open, high, low, close)."""
    idx = pd.bdate_range("2024-01-01", periods=len(bars))
    return pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1_000_000)


# A textbook bullish signal: a down bar, then a bar that engulfs its whole
# range, closes up, and leaves a lower wick bigger than its upper wick.
lead = (100.0, 100.2, 99.0, 99.2)              # down bar, so `setup` itself is not a bearish signal
setup = (100.0, 101.0, 98.0, 98.5)             # prior down bar
bull_bar = (98.4, 101.5, 96.0, 100.5)          # dn wick 2.4 vs up wick 1.0
df = frame([lead, setup, bull_bar])
bull, bear = _outside_bar_signals(df, 1.0, 0.0, False, False)
assert list(bull) == [False, False, True], list(bull)
assert not bear.any()
print("1. bullish outside bar with dominant lower wick  OK")

# Same bar, but the wick ratio is raised above what it actually has
# (2.4 / 1.0 = 2.4): 2.0 still passes, 3.0 must not.
assert _outside_bar_signals(df, 2.0, 0.0, False, False)[0].iloc[-1]
assert not _outside_bar_signals(df, 3.0, 0.0, False, False)[0].iloc[-1]
print("2. wick_ratio gate  OK")

# Prior candle was UP, so there is no down move to reverse -- the Pine
# prevOKBull filter rejects it even though the bar itself qualifies.
df_prev_up = frame([(99.0, 100.0, 98.5, 99.5), (98.0, 101.0, 97.5, 100.0), (99.9, 101.5, 96.0, 101.2)])
assert not _outside_bar_signals(df_prev_up, 1.0, 0.0, False, False)[0].any()
print("3. prior-candle direction filter  OK")

# A doji prior bar is rejected by default and accepted with allow_doji_prev.
df_doji = frame([(99.0, 100.0, 98.5, 99.5), (99.0, 101.0, 98.0, 99.0), (98.9, 101.5, 96.0, 100.5)])
assert not _outside_bar_signals(df_doji, 1.0, 0.0, False, False)[0].any()
assert _outside_bar_signals(df_doji, 1.0, 0.0, False, True)[0].iloc[-1]
print("4. allow_doji_prev  OK")

# Not an outside bar: the high does not exceed the prior high.
df_inside = frame([lead, setup, (98.4, 100.9, 96.0, 100.5)])
assert not _outside_bar_signals(df_inside, 1.0, 0.0, False, False)[0].any()
print("5. requires higher high AND lower low  OK")

# Mirror image: an up bar followed by a bearish outside bar with a dominant
# upper wick is the exit signal, not an entry.
bear_seq = frame([
    (99.0, 100.0, 98.5, 99.5),
    (98.5, 101.0, 98.0, 100.5),      # prior up bar
    (100.6, 104.0, 97.5, 99.0),      # up wick 3.4 vs dn wick 1.5
])
signals = outside_bar_reversal(bear_seq)
assert list(signals["exit"]) == [False, False, True], list(signals["exit"])
assert not signals["entry"].any()
print("6. bearish mirror signal drives the exit column  OK")

# The RSI gate can only remove entries, never add them: 30 flat bars then
# the bullish setup leaves RSI(21) right at the midline-ish region, so just
# assert the subset relationship rather than a specific RSI value.
long_df = frame([(100.0, 100.5, 99.5, 100.0)] * 29 + [lead, setup, bull_bar])
plain = outside_bar_reversal(long_df)["entry"]
gated = outside_bar_rsi(long_df)["entry"]
assert plain.any()
assert (gated & ~plain).sum() == 0
print("7. RSI gate is a strict subset of the unfiltered entries  OK")

print("\nall outside-bar checks passed")
