"""Swing points, BOS/CHoCH and FVGs in smc_regime.structure.

The assertion that matters most is timing. A pivot needs bars on BOTH
sides, so it cannot be known on the bar it happened -- only `length` bars
later. A detector that reports it in place looks identical in every summary
statistic and quietly lets a strategy trade a level it could not have drawn.
"""
import pandas as pd

from smc_regime import structure as st


def frame(highs, lows, closes=None, opens=None):
    n = len(highs)
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1] * n},
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )


# --- swing points are stamped where they are CONFIRMED -------------------
# One clean peak at position 5, with two bars either side (length=2).
highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
lows = [h - 2 for h in highs]
df = frame(highs, lows)
sw = st.swing_points(df, length=2)

peak_bar = df.index[5]
confirm_bar = df.index[7]
assert not sw.loc[peak_bar, "high_confirmed"], "a pivot cannot be known on its own bar"
assert sw.loc[confirm_bar, "high_confirmed"], "pivot should confirm `length` bars later"
assert sw.loc[confirm_bar, "high_price"] == 20
assert sw.loc[confirm_bar, "high_time"] == peak_bar
assert sw["high_confirmed"].sum() == 1

# Truncating the series after the confirming bar changes nothing about it --
# the same guarantee, stated as a property rather than an index arithmetic.
assert st.swing_points(df.loc[:confirm_bar], length=2).loc[confirm_bar, "high_confirmed"]

# A flat top still counts: requiring a strict maximum would drop double tops,
# which are exactly the levels price is respecting.
flat = frame([10, 11, 20, 20, 11, 10, 9], [8] * 7)
assert st.swing_points(flat, length=2)["high_confirmed"].any()

# --- BOS continues, CHoCH reverses ---------------------------------------
# Up to a peak, pull back to a trough, break the peak (BOS up), then break
# the trough (CHoCH down, because the trend was up).
highs = [10, 11, 12, 20, 14, 13, 12, 11, 15, 22, 18, 14, 10, 6, 5]
lows = [8, 9, 10, 18, 12, 11, 5, 9, 13, 20, 16, 12, 8, 4, 3]
closes = [9, 10, 11, 19, 13, 12, 6, 10, 14, 21, 17, 13, 9, 5, 4]
df = frame(highs, lows, closes)
ev = st.structure_events(df, length=2)

assert ev["bos_bullish"].any(), "breaking a confirmed swing high should be a BOS"
assert ev["choch_bearish"].any(), "breaking the swing low after an uptrend is a CHoCH"
first_bos = ev.index[ev["bos_bullish"]][0]
first_choch = ev.index[ev["choch_bearish"]][0]
assert first_bos < first_choch, "the trend has to be established before it can change"
assert ev.loc[first_bos, "trend"] == st.BULLISH
assert ev.loc[first_choch, "trend"] == st.BEARISH
# The same bar is never both.
assert not (ev["bos_bullish"] & ev["choch_bullish"]).any()
assert not (ev["bos_bearish"] & ev["choch_bearish"]).any()
# A level is consumed by the break that uses it, so one swing high cannot
# fire a BOS on every later bar.
assert ev["bos_bullish"].sum() <= ev.index.size // 3

# --- fair value gaps ------------------------------------------------------
# Bar 2 gaps clean of bar 0: low 30 above high 12, so 12..30 never traded.
highs = [12, 25, 40, 38, 36, 20, 15]
lows = [8, 20, 30, 28, 26, 10, 5]
df = frame(highs, lows)
fvg = st.fair_value_gaps(df)
assert fvg["fvg_bullish"].iloc[2], "low[2] above high[0] is a bullish imbalance"
assert not fvg["fvg_bullish"].iloc[:2].any()

# Price coming back into the zone registers as a touch while it stays open,
# and trading clean through the far side removes it.
assert fvg["in_bullish_fvg"].iloc[5], "bar 5 trades back down into the 12..30 gap"
assert fvg["open_bullish"].iloc[6] == 0, "a fully traded-through gap is no longer a gap"

# Bearish mirror.
down = frame([40, 30, 15, 18, 20], [30, 22, 10, 12, 14])
assert st.fair_value_gaps(down)["fvg_bearish"].iloc[2]

# A minimum size in ATR terms suppresses the tick-wide ones.
assert st.fair_value_gaps(df, min_size_atr=50.0)["fvg_bullish"].sum() == 0

print("all structure tests passed")
