"""Every trade records WHICH mechanism closed it.

Nothing recorded this before, so "do stops cut the winners short, or does
the strategy's own exit rule?" could only be argued -- the trade list looked
identical whichever fired. Stop exits are split into "stop" and "stop_gap"
because a gapped stop is a different event: it fills below the stop level,
and it is what drives the loss tail.

Check 7 is the regression that matters beyond the new field:
cross_validate._finalized_trades re-derived the open position itself and
double-counted it once run_backtest started marking the survivor to market.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import Trade, run_backtest

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


def sig(df, entry_at=0, exit_at=None):
    s = pd.DataFrame({"entry": [False] * len(df), "exit": [False] * len(df)}, index=df.index)
    s.iloc[entry_at, s.columns.get_loc("entry")] = True
    if exit_at is not None:
        s.iloc[exit_at, s.columns.get_loc("exit")] = True
    return s


# a gap straight through the stop
gapped = bars([(100, 100, 100, 100), (80, 82, 78, 79)])
t = run_backtest(gapped, sig(gapped), stop_loss_pct=10.0)[0]
check("1. a gapped stop reads stop_gap", t.exit_reason == "stop_gap", f"  ({t.exit_reason})")

# opens above the stop, trades down through it intraday
intraday = bars([(100, 100, 100, 100), (99, 99, 85, 86)])
t2 = run_backtest(intraday, sig(intraday), stop_loss_pct=10.0)[0]
check("2. an intraday stop reads stop", t2.exit_reason == "stop", f"  ({t2.exit_reason})")

# the strategy's own exit fires
plain = bars([(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102)])
t3 = run_backtest(plain, sig(plain, 0, 2))[0]
check("3. a strategy exit reads signal", t3.exit_reason == "signal", f"  ({t3.exit_reason})")

# the hold limit forces the close
t4 = run_backtest(plain, sig(plain, 0), max_hold_bars=2)[0]
check("4. a forced close reads max_hold", t4.exit_reason == "max_hold", f"  ({t4.exit_reason})")

# never exits: marked to market at the last bar
t5 = run_backtest(plain, sig(plain, 0))[-1]
check("5. an open position reads open and stays flagged",
      t5.exit_reason == "open" and t5.is_open, f"  ({t5.exit_reason}, is_open={t5.is_open})")

# .stopped groups both stop flavours and excludes the rest
check("6. .stopped covers both stop kinds and nothing else",
      t.stopped and t2.stopped and not t3.stopped and not t4.stopped and not t5.stopped)

# the default keeps old positional construction meaning what it did
legacy = Trade(plain.index[0], plain.index[2], 100.0, 110.0)
check("7. a positionally-built Trade still defaults to signal",
      legacy.exit_reason == "signal" and not legacy.is_open and not legacy.stopped)

# cross_validate must not re-append a trade run_backtest already returned
from smc_regime.cross_validate import _finalized_trades
from smc_regime.strategies import STRATEGIES

import numpy as np
rng = np.random.default_rng(3)
n = 300
close = 100 * np.cumprod(1 + rng.normal(0.002, 0.012, n))
idx = pd.date_range("2024-01-01", periods=n, freq="B")
real = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                     "Close": close, "Volume": [1e6] * n}, index=idx)
ok = True
for strat in ("rsi_dip_recovery", "macd"):
    base = run_backtest(real, STRATEGIES[strat](real))
    fin = _finalized_trades(real, strat)
    if len(fin) != len(base):
        ok = False
    seen = {}
    for tr in fin:
        key = (tr.entry_date, round(tr.entry_price, 6))
        seen[key] = seen.get(key, 0) + 1
    if any(v > 1 for v in seen.values()):
        ok = False
check("8. _finalized_trades does not double-count the open position", ok)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
