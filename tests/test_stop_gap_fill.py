"""A stop fills at the open when the bar gapped through it.

run_backtest used to fill every stop at stop_price the moment a bar's Low
touched it. That treats a stop as a limit order resting at the level, but a
stop is an instruction to sell at the market once the level trades -- on a
gap down the level is never available and the fill is the open. The old
behaviour handed stops their exact price in precisely the scenario they are
worst at, which is the one that matters when weighing a stop against buying
a put.

On the 415-ticker 1d universe, holding a name already up 20% over 60 bars
behind an 8% stop, 17.8% of triggering bars opened BELOW the stop, filling
2.21% worse than the stop on average and 7.61% worse at p95.

Checks 1-2 are the two branches; 3 pins that the old behaviour is actually
gone rather than coincidentally matching; 4-5 keep slippage and the
intraday path intact.
"""
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import run_backtest

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


def bars(rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(
        {
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[2] for r in rows],
            "Close": [r[3] for r in rows],
            "Volume": [1000] * len(rows),
        },
        index=idx,
    )


def sig(df, entry_at=0):
    s = pd.DataFrame({"entry": [False] * len(df), "exit": [False] * len(df)}, index=df.index)
    s.iloc[entry_at, s.columns.get_loc("entry")] = True
    return s


# Enter at bar 0's close of 100, stop 10% -> 90.
# Bar 1 GAPS: opens at 80, never trades near 90.
gapped = bars([(100, 100, 100, 100), (80, 82, 78, 79)])
t = run_backtest(gapped, sig(gapped), stop_loss_pct=10.0)[0]
check("1. a gap through the stop fills at the open, not the stop",
      abs(t.exit_price - 80.0) < 1e-9, f"  (exit_price={t.exit_price})")

# Bar 1 opens ABOVE the stop and trades down through it intraday.
# The stop was reachable, so it fills at the stop.
intraday = bars([(100, 100, 100, 100), (99, 99, 85, 86)])
t2 = run_backtest(intraday, sig(intraday), stop_loss_pct=10.0)[0]
check("2. an intraday break still fills at the stop level",
      abs(t2.exit_price - 90.0) < 1e-9, f"  (exit_price={t2.exit_price})")

# The old code filled the gapped case at 90 (a -10% trade). The point of the
# change is that the gap is MORE expensive than the stop level promised.
check("3. the gapped trade is worse than the stop level implied",
      t.exit_price < 90.0 and t.return_pct < -10.0,
      f"  (return_pct={t.return_pct:.2f}%)")

# Slippage applies to whichever price was actually filled. Guarded on the
# engine advertising the parameter: the gap fill and per-fill slippage are
# separate changes and reached the default branch separately, so this file
# has to run on a tree that has one and not the other.
if "slippage_pct" in inspect.signature(run_backtest).parameters:
    t3 = run_backtest(gapped, sig(gapped), stop_loss_pct=10.0, slippage_pct=0.10)[0]
    check("4. slippage is charged on the gap fill",
          abs(t3.exit_price - 80.0 * (1 - 0.001)) < 1e-9, f"  (exit_price={t3.exit_price})")
else:
    print("4. slippage is charged on the gap fill  SKIPPED (engine takes no slippage_pct)")

# A bar whose open is exactly the stop is not a gap; it fills at the stop.
at_stop = bars([(100, 100, 100, 100), (90, 90, 88, 89)])
t4 = run_backtest(at_stop, sig(at_stop), stop_loss_pct=10.0)[0]
check("5. an open exactly at the stop fills at the stop",
      abs(t4.exit_price - 90.0) < 1e-9, f"  (exit_price={t4.exit_price})")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
