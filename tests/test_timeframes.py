"""Ladder aggregation in smc_regime.timeframes, on a synthetic RTH session.

The property that matters most here is the absence of look-ahead: a higher
rung is only allowed to inform a base bar once that rung's bar has closed.
Getting it wrong is invisible in a backtest's output and inflates every
number, so it is asserted directly rather than inferred from results.
"""
import pandas as pd

from smc_regime import indicators as ind
from smc_regime import patterns as pat
from smc_regime import timeframes as tf

BARS_PER_SESSION = 26  # 390-minute RTH / 15m


def session_index(day):
    return pd.date_range(f"{day} 09:30", periods=BARS_PER_SESSION, freq="15min", tz=tf.MARKET_TZ)


def make_intraday(days=("2024-03-04", "2024-03-05"), opening_range=4.0, normal_range=0.4):
    """Two RTH sessions of 15m bars. The 09:30 bar is deliberately given a
    much wider range than the rest, mirroring the real intraday volatility
    smile that slot_atr exists to handle."""
    rows, index = [], []
    price = 100.0
    for day in days:
        for i, ts in enumerate(session_index(day)):
            rng = opening_range if i == 0 else normal_range
            rows.append((price, price + rng, price - rng, price + rng / 4, 1_000))
            price += rng / 4
            index.append(ts)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(index))


df = make_intraday()
assert len(df) == 2 * BARS_PER_SESSION

# --- the ladder's shape, per session ------------------------------------
h1 = tf.session_resample(df, "60min")
h4 = tf.session_resample(df, "240min")
d1 = tf.session_resample(df, "1D")

per_session = lambda x: x.groupby(tf._session_dates(x.index)).size().unique().tolist()
assert per_session(h1) == [7], per_session(h1)     # 6 full hours + a 30-min stub
assert per_session(h4) == [2], per_session(h4)     # 4h then a 2.5h bar
assert per_session(d1) == [1], per_session(d1)

# The raggedness is recorded, not hidden: the last hour of the session is
# built from 2 base bars where a full one uses 4.
first_session_h1 = h1[tf._session_dates(h1.index) == pd.Timestamp("2024-03-04").date()]
assert list(first_session_h1["source_bars"]) == [4, 4, 4, 4, 4, 4, 2]
assert list(h4["source_bars"])[:2] == [16, 10]
assert list(d1["source_bars"]) == [26, 26]

# Sessions never merge: the overnight gap always starts a new bar.
assert len(d1) == 2
assert h4.index[0].date() != h4.index[2].date()

# --- bar_end is when the bar became knowable -----------------------------
# 15m bars, so a bar that last consumed 13:15 finished at 13:30.
assert h4["bar_end"].iloc[0] == pd.Timestamp("2024-03-04 13:30", tz=tf.MARKET_TZ)
assert (h4["bar_end"] > h4.index).all()

# --- no look-ahead --------------------------------------------------------
# Mark the first 4h bar True and check when the base series is allowed to
# see it: never while that bar is still forming, first at its close.
flag = pd.Series(False, index=h4.index)
flag.iloc[0] = True
aligned = tf.align_to_base(flag, h4["bar_end"], df.index)

inside = pd.Timestamp("2024-03-04 12:00", tz=tf.MARKET_TZ)
at_close = pd.Timestamp("2024-03-04 13:30", tz=tf.MARKET_TZ)
assert not aligned.loc[inside], "a base bar read a 4h bar it was still inside"
assert aligned.loc[at_close], "the 4h bar's own close should be able to see it"
assert not aligned.iloc[0]  # nothing has closed yet at the very first bar
assert aligned.dtype == bool

# Held for the rest of the session -- the afternoon 4h bar only closes at
# 16:00, after the final 15m bar, so nothing supersedes it until day two.
day_one_after_close = aligned.loc[at_close:].loc[: pd.Timestamp("2024-03-04 15:45", tz=tf.MARKET_TZ)]
assert day_one_after_close.all()
assert not aligned.loc[pd.Timestamp("2024-03-05 09:30", tz=tf.MARKET_TZ)]

# Same guarantee through the public entry point.
state = tf.higher_timeframe_state(df, "240min", detector=lambda d: pd.DataFrame(
    {"bullish": pd.Series([True] + [False] * (len(d) - 1), index=d.index),
     "bearish": pd.Series(False, index=d.index)}))
assert not state.loc[inside, "bullish"]
assert state.loc[at_close, "bullish"]

# --- rungs coarser than a session are refused, not mis-binned ------------
try:
    tf.session_resample(df, "W")
except ValueError as exc:
    assert "coarser than a session" in str(exc)
else:
    raise AssertionError("a weekly rung should have been rejected")

# --- ladder_state assembles per-rung columns and a count -----------------
ladder = tf.ladder_state(df, rules=("60min", "240min"))
assert list(ladder.columns) == [
    "60min_bullish", "60min_bearish", "240min_bullish", "240min_bearish",
    "rungs_bullish", "rungs_bearish",
]
assert len(ladder) == len(df)
assert ladder["rungs_bullish"].max() <= 2

# --- slot-normalised ATR --------------------------------------------------
# On daily bars every bar shares one slot, so this must be exactly ind.atr.
daily = pd.DataFrame(
    {"Open": [100, 101, 102, 103, 104], "High": [101, 102, 103, 104, 105],
     "Low": [99, 100, 101, 102, 103], "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
     "Volume": [1] * 5},
    index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
)
assert (pat.slot_atr(daily) - ind.atr(daily)).abs().max() < 1e-12

# Intraday, the wide 09:30 bar must not inflate the baseline every other
# bar is measured against -- which is exactly what a blended ATR does.
slot = pat.slot_atr(df)
blended = pat.slot_atr(df, slot_normalized=False)
opening = df.index.time == pd.Timestamp("09:30").time()
midday = df.index.time == pd.Timestamp("12:00").time()
assert slot[opening].iloc[-1] > blended[opening].iloc[-1]
assert slot[midday].iloc[-1] < blended[midday].iloc[-1]

# --- bar_direction is dense where reversal_signals is sparse -------------
direction = pat.bar_direction(df)
reversal = pat.reversal_signals(df)
assert direction["bullish"].sum() > reversal["bullish"].sum()

# Daily bars are not all stamped at the same time of day -- Yahoo uses the
# session open in UTC, so a DST change gives one series two distinct stamps.
# Slot grouping must not engage there: it split SPY's daily series in two and
# moved its backtest average before this guard existed.
dst_daily = daily.copy()
dst_daily.index = pd.DatetimeIndex(
    ["2024-03-06 14:30", "2024-03-07 14:30", "2024-03-08 14:30",
     "2024-03-11 13:30", "2024-03-12 13:30"], tz="UTC"
)
assert len(set(dst_daily.index.time)) == 2, "the fixture must actually straddle a DST change"
assert (pat.slot_atr(dst_daily) - ind.atr(dst_daily)).abs().max() < 1e-12

print("all timeframe tests passed")
