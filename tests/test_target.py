"""run_backtest's optional profit target (target_pct / target_pct_series).

Added for ChartLab's Strategy Lab (fixed R-multiple exits such as the 3R
target of the swing-failure research). It must change NOTHING for callers
that do not pass it -- check 6 runs every registered strategy through the
engine with and without an explicit target=None and compares the trades.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import backtest_strategy, run_backtest
from smc_regime.strategies import STRATEGIES

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


def bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(
        {"Open": [r[0] for r in rows], "High": [r[1] for r in rows],
         "Low": [r[2] for r in rows], "Close": [r[3] for r in rows],
         "Volume": [1000] * len(rows)}, index=idx)


def sig(df, entry_at=0):
    s = pd.DataFrame({"entry": [False] * len(df), "exit": [False] * len(df)}, index=df.index)
    s.iloc[entry_at, 0] = True
    return s


# 1. target touched intrabar fills AT the target
df = bars([(100, 101, 99, 100), (100, 109, 99, 104), (104, 111, 103, 110), (110, 112, 109, 111)])
t = run_backtest(df, sig(df), target_pct=10)
check("1 intrabar target fills at the target", len(t) == 1 and abs(t[0].exit_price - 110) < 1e-9 and t[0].exit_reason == "target",
      f"  ({t[0].exit_price if t else None}, {t[0].exit_reason if t else None})")

# 2. a gap above the target fills at the OPEN (better than the target)
df = bars([(100, 101, 99, 100), (100, 102, 99, 101), (115, 118, 114, 116)])
t = run_backtest(df, sig(df), target_pct=10)
check("2 gap above the target fills at the open", len(t) == 1 and abs(t[0].exit_price - 115) < 1e-9 and t[0].exit_reason == "target")

# 3. a bar touching both the stop and the target counts as the stop (pessimistic)
df = bars([(100, 101, 99, 100), (100, 112, 94, 100)])
t = run_backtest(df, sig(df), stop_loss_pct=5, target_pct=10)
check("3 stop and target on one bar -> stop", len(t) == 1 and t[0].exit_reason == "stop" and abs(t[0].exit_price - 95) < 1e-9)

# 4. per-entry series: the value at the entry bar is fixed for the trade
df = bars([(100, 101, 99, 100), (100, 104, 99, 103), (103, 106, 102, 105), (105, 109, 104, 108)])
ser = pd.Series([5.0, 50.0, 50.0, 50.0], index=df.index)
t = run_backtest(df, sig(df), target_pct_series=ser)
check("4 target series read at the entry bar", len(t) == 1 and abs(t[0].exit_price - 105) < 1e-9 and t[0].exit_date == df.index[2])

# 5. a strategy emitting target_pct gets it through backtest_strategy
STRATEGIES["_t_target"] = lambda d: sig(d).assign(target_pct=10.0)
df = bars([(100, 101, 99, 100), (100, 111, 99, 110), (110, 112, 109, 111)])
t = backtest_strategy(df, "_t_target")
del STRATEGIES["_t_target"]
check("5 backtest_strategy honours a target_pct column", len(t) == 1 and t[0].exit_reason == "target")

# 6. regression: no target passed -> every registered strategy produces the same trades as before
rng = np.random.default_rng(7)
close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 900)))
openp = close * (1 + rng.normal(0, 0.004, 900))
high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.008, 900)))
low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.008, 900)))
df = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": rng.integers(1e5, 1e6, 900).astype(float)},
                  index=pd.bdate_range("2020-01-01", periods=900))
same, n = 0, 0
for name in STRATEGIES:
    try:
        a = backtest_strategy(df, name, stop_loss_pct=8, max_hold_bars=30)
        b = backtest_strategy(df, name, stop_loss_pct=8, max_hold_bars=30, target_pct=None, target_pct_series=None)
    except Exception as e:
        print(f"   {name}: skipped ({type(e).__name__})")
        continue
    n += 1
    same += [(x.entry_date, x.exit_date, x.exit_price, x.exit_reason) for x in a] == [(x.entry_date, x.exit_date, x.exit_price, x.exit_reason) for x in b]
check(f"6 no target -> identical trades for every strategy ({same}/{n})", same == n and n > 0)

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
