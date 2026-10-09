"""Connors' 2-period RSI: the two registered variants do what they claim.

These went onto default on the strength of a measurement (SPY replication
plus a positive excess) with no test pinning their behaviour, which is out
of step with every other strategy added here. This closes that.

The assertions are behavioural rather than arithmetic -- entries are
checked against ind.rsi()'s own output rather than against hand-computed
RSI values, so the test pins the STRATEGY and not a second implementation
of the indicator that could drift from the first.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import indicators as ind
from smc_regime.backtest import backtest_strategy
from smc_regime.strategies import STRATEGIES, rsi2_connors, rsi2_connors_prior_high

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


def frame(closes, highs=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c * 1.01 if highs is None else np.asarray(highs, float),
         "Low": c * 0.99, "Close": c, "Volume": [1e6] * len(c)}, index=idx)


# a deliberate shape: flat, two sharp down days, then a sharp recovery
shape = [100, 100, 100, 94, 88, 89, 96, 103, 104, 104, 104, 97, 92, 99, 105, 105]
df = frame(shape)
r2 = ind.rsi(df["Close"], 2)

# --- 1-2: entries are exactly the oversold bars ------------------------
sig = rsi2_connors(df)
check("1. rsi2 enters exactly where RSI(2) < oversold",
      bool((sig["entry"] == (r2 < 10.0).fillna(False)).all()),
      f"  ({int(sig['entry'].sum())} entries)")

sig_ph = rsi2_connors_prior_high(df)
check("2. both variants share the same entry rule",
      bool((sig_ph["entry"] == sig["entry"]).all()))

# --- 3-4: the two exit rules -------------------------------------------
want_exit = ((r2 > 80.0) & (r2.shift(1) <= 80.0)).fillna(False)
check("3. rsi2 exits on the up-cross of exit_rsi, not merely above it",
      bool((sig["exit"] == want_exit).all()), f"  ({int(sig['exit'].sum())} exits)")

want_ph = (df["Close"] > df["High"].shift(1)).fillna(False)
check("4. rsi2_prior_high exits on a close above the prior bar's high",
      bool((sig_ph["exit"] == want_ph).all()), f"  ({int(sig_ph['exit'].sum())} exits)")

# --- 5: neither carries a stop ----------------------------------------
check("5. neither variant emits stop_pct, so both run stopless",
      "stop_pct" not in sig and "stop_pct" not in sig_ph)

# --- 6: the threshold actually moves the signal count -----------------
rng = np.random.default_rng(11)
walk = frame(100 * np.cumprod(1 + rng.normal(0.0004, 0.014, 900)))
counts = [int(rsi2_connors(walk, oversold=t)["entry"].sum()) for t in (2, 5, 10, 20, 30)]
check("6. a looser oversold threshold never yields fewer entries",
      all(a <= b for a, b in zip(counts, counts[1:])) and counts[0] < counts[-1],
      f"  (thresholds 2/5/10/20/30 -> {counts})")

# --- 7: the point of a 2-bar lookback is reactivity -------------------
short, long_ = ind.rsi(walk["Close"], 2), ind.rsi(walk["Close"], 14)
check("7. RSI(2) reaches extremes RSI(14) does not on the same series",
      (short < 10).sum() > (long_ < 10).sum() and (short > 90).sum() > (long_ > 90).sum(),
      f"  (<10: {int((short<10).sum())} vs {int((long_<10).sum())}; "
      f">90: {int((short>90).sum())} vs {int((long_>90).sum())})")

# --- 8: the prior-high exit is the tighter one ------------------------
a = [t for t in backtest_strategy(walk, "rsi2", interval="1d") if not t.is_open]
b = [t for t in backtest_strategy(walk, "rsi2_prior_high", interval="1d") if not t.is_open]
pos = {s: i for i, s in enumerate(walk.index)}
hold_a = np.median([pos[t.exit_date] - pos[t.entry_date] for t in a])
hold_b = np.median([pos[t.exit_date] - pos[t.entry_date] for t in b])
check("8. the prior-high exit holds for fewer bars than the RSI exit",
      hold_b < hold_a, f"  (median {hold_b:.0f} vs {hold_a:.0f} bars)")

# --- 9: both are registered and reachable through the engine ----------
check("9. both are registered under their documented names",
      "rsi2" in STRATEGIES and "rsi2_prior_high" in STRATEGIES
      and STRATEGIES["rsi2"] is rsi2_connors
      and STRATEGIES["rsi2_prior_high"] is rsi2_connors_prior_high)
check("10. both produce trades through backtest_strategy",
      len(a) > 0 and len(b) > 0, f"  ({len(a)} and {len(b)} closed trades)")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
