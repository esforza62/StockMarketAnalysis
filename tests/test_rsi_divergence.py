"""Checks for the RSI-divergence signals in smc_regime.strategies.

Synthetic bars throughout -- these run without a Tiingo key. Each fixture is
a price path with a known answer: a first low, a bounce, then a second,
marginally lower low reached either gently (momentum holds up -> divergence)
or steeply (momentum makes a new low too -> no divergence).
"""
import numpy as np
import pandas as pd

from smc_regime import indicators as ind
from smc_regime.backtest import run_backtest
from smc_regime.strategies import rsi_divergence, rsi_divergence_5d, rsi_divergence_masks


def frame(values):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"Open": values, "High": values + 0.2, "Low": values - 0.2, "Close": values, "Volume": 1e6},
        index=pd.bdate_range("2024-01-01", periods=len(values)),
    )


def path(second_leg_from, second_leg_to=87.4):
    """Decline to a low at bar 20, bounce, then decline again to a lower low
    at bar 32. The alternating wiggle keeps both gains and losses present in
    every RSI window -- a monotone leg makes RSI exactly 0 (or undefined on
    the mirrored series) and the comparison stops meaning anything."""
    base = np.array(
        list(np.linspace(100, 88, 21)) + list(np.linspace(89.5, 94, 5)) + list(np.linspace(second_leg_from, second_leg_to, 8))
    )
    wiggle = np.array([0.0 if i in (20, len(base) - 1) else (0.6 if i % 2 else -0.6) for i in range(len(base))])
    return base + wiggle


# 1. Gentle second leg: price makes a lower low, RSI does not -> bullish divergence.
gentle = frame(path(93))
bull, bear = rsi_divergence_masks(gentle, 14, 20)
r = ind.rsi(gentle["Close"], 14)
assert list(np.where(bull.to_numpy())[0]) == [32], np.where(bull.to_numpy())[0]
assert gentle["Low"].iloc[32] < gentle["Low"].iloc[20], "fixture should make a lower low"
assert r.iloc[32] > r.iloc[20], "fixture should make a higher RSI low"
assert not bear.any()
print("1. bullish divergence: lower low, higher RSI  OK")

# 2. The mirrored price path must produce the bearish signal on the same bar.
mirrored = frame(200 - path(93))
m_bull, m_bear = rsi_divergence_masks(mirrored, 14, 20)
assert list(np.where(m_bear.to_numpy())[0]) == [32], np.where(m_bear.to_numpy())[0]
assert not m_bull.any()
print("2. bearish divergence is the mirror image  OK")

# 3. Negative control: same lower low, but reached steeply enough that RSI
#    makes a lower low too -- that is confirmation, not divergence.
steep = frame(path(92, 82.0))
steep_bull, _ = rsi_divergence_masks(steep, 14, 20)
r_steep = ind.rsi(steep["Close"], 14)
assert steep["Low"].iloc[32] < steep["Low"].iloc[20]
assert r_steep.iloc[32] < r_steep.iloc[20]
assert not steep_bull.any(), "RSI confirming the low must not read as divergence"
print("3. lower low WITH lower RSI is not divergence  OK")

# 4. No lookahead: signals on a truncated series must match the same bars
#    computed with the later bars present.
long_path = frame(np.concatenate([path(93), path(93) + 5]))
full = rsi_divergence_masks(long_path, 14, 20)[0]
trunc = rsi_divergence_masks(long_path.iloc[:40], 14, 20)[0]
assert (full.iloc[:40] == trunc).all(), "signal changed when future bars were added"
print("4. no lookahead  OK")

# 5. The lookback bounds which prior low counts: with a window shorter than
#    the gap between the two lows, there is no prior low left to diverge from.
assert not rsi_divergence_masks(gentle, 14, 5)[0].any()
print("5. lookback window bounds the comparison  OK")

# 6. rsi_divergence_5d exits five bars after entry, not on the mirror signal.
holds = []
signals = rsi_divergence_5d(long_path)
pos = pd.Series(range(len(long_path)), index=long_path.index)
for trade in run_backtest(long_path, signals):
    holds.append(pos[trade.exit_date] - pos[trade.entry_date])
assert holds, "fixture should generate at least one trade"
assert set(holds) == {5}, holds
assert rsi_divergence(long_path)["exit"].equals(rsi_divergence_masks(long_path)[1]), "default exit is the mirror signal"
print("6. rsi_divergence_5d holds exactly five bars  OK")

print("\nall RSI-divergence checks passed")
