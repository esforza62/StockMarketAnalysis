"""Swing failure: sweep a confirmed swing low, close back above it.

The failure mode this file exists to catch is LOOKAHEAD. A pivot low at
bar i is only a pivot once `right` further bars have failed to beat it, so
it cannot be known until bar i + right. A backtest that scans for sweeps
against the raw centred rolling minimum trades a level the market had not
identified yet, and a liquidity-sweep strategy built that way looks
superb for entirely fake reasons.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import indicators as ind
from smc_regime.backtest import backtest_strategy, run_backtest
from smc_regime.strategies import STRATEGIES, swing_failure, swing_failure_delayed

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


def frame(lows, highs=None, closes=None):
    lows = [float(x) for x in lows]
    highs = [x + 2 for x in lows] if highs is None else [float(x) for x in highs]
    closes = [x + 1 for x in lows] if closes is None else [float(x) for x in closes]
    idx = pd.date_range("2024-01-01", periods=len(lows), freq="D")
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                         "Close": closes, "Volume": [1000] * len(lows)}, index=idx)


# ---- the pivot helper must not see the future ---------------------------
df = frame([10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11])
sl = ind.swing_low(df, left=3, right=3)

check("1. a pivot low is invisible until `right` bars after it",
      sl.iloc[:8].isna().all())

check("2. and becomes visible exactly then, at i + right",
      sl.iloc[8] == 5.0 and sl.index[8] == df.index[8])

sh = ind.swing_high(frame([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9]), left=3, right=3)
check("3. the same holds for pivot highs", sh.iloc[:8].isna().all() and sh.iloc[8] == 17.0)

# The strongest form: truncating the series must not change any value that
# was already emitted. If a later bar could change an earlier reading, the
# earlier reading was using the future.
long_df = frame(list(np.round(100 + np.cumsum(np.random.default_rng(5).normal(0, 1, 200)), 3)))
full = ind.swing_low(long_df, 3, 3)
stable = True
for cut in (60, 100, 150):
    partial = ind.swing_low(long_df.iloc[:cut], 3, 3)
    if not partial.equals(full.iloc[:cut]):
        stable = False
check("4. truncating the data never changes an already-emitted pivot", stable)

# ---- the pattern fires where it should ----------------------------------
# Pivot low 5 at index 5, confirmed at index 8. Index 10 wicks to 4 (below
# 5) and closes at 7 (back above) -- a same-bar swing failure.
lows =  [10, 9, 8, 7, 6, 5, 6, 7, 8, 9,  4, 8, 9, 10, 11, 12]
closes= [11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 7, 9, 10, 11, 12, 13]
highs = [c + 1 for c in closes]
sfp = frame(lows, highs, closes)
sig = swing_failure(sfp, pivot_left=3, pivot_right=3)

check("5. the same-bar sweep-and-reclaim is an entry", bool(sig["entry"].iloc[10]))

check("6. and it is the only entry", sig["entry"].sum() == 1)

check("7. the stop sits below the swept low, not at a fixed distance",
      sig["stop_pct"].iloc[10] > 0
      and abs(closes[10] * (1 - sig["stop_pct"].iloc[10] / 100) - 4 * (1 - 0.25 / 100)) < 1e-6)

check("8. stop_pct is set only on the entry bar",
      sig["stop_pct"].notna().sum() == 1)

# A sweep that does NOT reclaim is not a swing failure.
lows2 =  [10, 9, 8, 7, 6, 5, 6, 7, 8, 9,  4, 3, 2, 1, 1, 1]
closes2= [11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 4.5, 3.5, 2.5, 1.5, 1.5, 1.5]
no_reclaim = swing_failure(frame(lows2, [c + 1 for c in closes2], closes2), 3, 3)
check("9. a sweep with no reclaim is not an entry", no_reclaim["entry"].sum() == 0)

# ---- the two variants genuinely differ ----------------------------------
# Sweep at indices 10-11 (low under the 5.0 level, close still below it),
# then index 12 trades ENTIRELY above the level -- low 5.5, close 7.0 -- so
# it reclaims without sweeping on that bar. Same-bar must reject it; the
# delayed variant, one bar after the sweep, must take it.
#
# An earlier version of this fixture put index 12's low at 4.5, which is
# below the 5.0 level: that bar swept AND reclaimed on itself, so the
# same-bar variant was right to take it and the test was wrong, not the
# code.
lows3 =  [10, 9, 8, 7, 6, 5, 6, 7, 8, 9,   4,   4.2, 5.5, 9, 10, 11]
closes3= [11, 10, 9, 8, 7, 6, 7, 8, 9, 10, 4.5, 4.8, 7.0, 10, 11, 12]
delayed_df = frame(lows3, [c + 1 for c in closes3], closes3)

same_bar = swing_failure(delayed_df, 3, 3)
later = swing_failure_delayed(delayed_df, 3, 3, reclaim_bars=3)

check("10. the same-bar variant rejects a reclaim that arrives later",
      same_bar["entry"].sum() == 0)

check("11. the delayed variant takes it", bool(later["entry"].iloc[12]))

# Registration, not the registry's size: asserting a total count makes
# every future strategy break this test for no reason.
from smc_regime.export_dashboard_data import STRATEGY_NAMES

_SWING = ["swing_failure", "swing_failure_delayed", "swing_failure_chop_filter"]
check("12. each variant is registered as its own strategy",
      all(name in STRATEGIES for name in _SWING))

check("12b. and each has a display label for the desk",
      all(name in STRATEGY_NAMES for name in _SWING))

# ---- the stop is actually wired into the engine -------------------------
rng = np.random.default_rng(3)
n = 600
close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, n)))
big = pd.DataFrame({"Open": close, "High": close * (1 + abs(rng.normal(0, 0.008, n))),
                    "Low": close * (1 - abs(rng.normal(0, 0.008, n))), "Close": close,
                    "Volume": rng.integers(1e5, 1e6, n)},
                   index=pd.date_range("2024-01-01", periods=n, freq="D"))

signals = STRATEGIES["swing_failure_delayed"](big)
with_stop = [t.return_pct for t in backtest_strategy(big, "swing_failure_delayed") if not t.is_open]
without = [t.return_pct for t in run_backtest(big, signals[["entry", "exit"]]) if not t.is_open]

check("13. the strategy's own stop reaches the engine and truncates the worst loss",
      min(with_stop) > min(without))

# An explicit argument must still win, or a caller sweeping stop levels
# would silently be overridden by the strategy's own.
tight = [t.return_pct for t in backtest_strategy(big, "swing_failure_delayed", stop_loss_pct=0.5)
         if not t.is_open]
check("14. an explicit stop_loss_pct overrides the strategy's own",
      min(tight) > min(with_stop))

# And a strategy WITHOUT a stop_pct column must be unaffected by all this.
plain = STRATEGIES["rsi"](big)
check("15. strategies with no stop_pct column still run stopless",
      "stop_pct" not in plain)

# ---- the chop-filtered variant ------------------------------------------
from smc_regime.regime import classify_regime, confirmed_regime

chop_sig = STRATEGIES["swing_failure_chop_filter"](big)
base_sig = STRATEGIES["swing_failure_delayed"](big)

check("16. filtering only ever removes entries, never invents them",
      (chop_sig["entry"] & ~base_sig["entry"]).sum() == 0)

check("17. and it does remove some", chop_sig["entry"].sum() < base_sig["entry"].sum())

reg = confirmed_regime(classify_regime(big), confirm_bars=3)
entered_regimes = reg.loc[chop_sig["entry"][chop_sig["entry"]].index, "regime"]
check("18. no surviving entry is in a choppy regime",
      (entered_regimes == "choppy").sum() == 0)

check("19. a filtered-out entry drops its stop with it",
      chop_sig["stop_pct"].notna().sum() == chop_sig["entry"].sum())

# The filter must not reach forward: truncating the data cannot change an
# entry the strategy already emitted.
cut = 400
partial = STRATEGIES["swing_failure_chop_filter"](big.iloc[:cut])
check("20. truncating the data does not change earlier filtered entries",
      partial["entry"].equals(chop_sig["entry"].iloc[:cut]))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
