"""Every registered strategy, against the frames that actually turn up.

On 2026-10-10 a nightly run died twelve minutes in, after the 1d interval
had already produced 399,042 trades, because one ticker on a later interval
arrived with an object-dtype Low column and `swing_failure_delayed` reduces
with a groupby. A groupby reduction is the one numeric operation pandas
REFUSES on object dtype rather than limping along, so a defect every other
strategy tolerated silently aborted the run for the other 414 tickers.

The verification that missed it ran all 23 strategies against a single
clean float64 synthetic frame and reported no errors. That could not have
caught a dtype-dependent failure, so this file runs the whole registry
against the degenerate shapes instead: object dtype, all-NaN, a single bar,
and a series shorter than the indicator windows.

The contract is deliberately weak and that is the point. A strategy facing
a frame it cannot read must return an empty-or-False signal frame, not
raise: on a universe run, one bad ticker should cost one ticker.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.strategies import STRATEGIES

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


def clean(n=300):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(11)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n))), index=idx)
    return pd.DataFrame({"Open": c.shift(1).fillna(c.iloc[0]),
                         "High": c * 1.008, "Low": c * 0.992, "Close": c,
                         "Volume": rng.integers(1e6, 9e6, n).astype(float)}, index=idx)


base = clean()
OHLC = ["Open", "High", "Low", "Close"]

cases = {}
cases["object dtype OHLC"] = base.assign(**{c: base[c].astype(object) for c in OHLC})
cases["object dtype Low only"] = base.assign(Low=base["Low"].astype(object))
nan = base.copy()
nan[OHLC] = np.nan
cases["all-NaN OHLC"] = nan
cases["single bar"] = base.iloc[:1]
cases["ten bars"] = base.iloc[:10]
cases["empty, float dtype"] = base.iloc[:0]
cases["empty, object dtype"] = base.iloc[:0].assign(
    **{c: pd.Series(dtype=object) for c in OHLC})

print(f"{len(STRATEGIES)} strategies x {len(cases)} degenerate frames\n")
raised = {}
for case, frame in cases.items():
    bad = []
    for name, fn in STRATEGIES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = fn(frame)
            # must look like a signal frame
            if not isinstance(out, pd.DataFrame) or not {"entry", "exit"} <= set(out.columns):
                bad.append(f"{name}(shape)")
        except Exception as e:
            bad.append(f"{name}({type(e).__name__})")
    raised[case] = bad
    check(f"  {case:<24}", not bad,
          "" if not bad else "  -> " + ", ".join(bad[:4]) + ("..." if len(bad) > 4 else ""))

# The specific regression: the exact failure that killed the run.
obj = cases["object dtype OHLC"]
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = STRATEGIES["swing_failure_delayed"](obj)
    ok = isinstance(got, pd.DataFrame)
except Exception as e:
    ok = False
    got = type(e).__name__
check("regression: swing_failure_delayed survives object dtype",
      ok, "" if ok else f"  ({got})")

# and it must give the SAME answer it would on float, not merely survive
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    a = STRATEGIES["swing_failure_delayed"](base)["entry"]
    b = STRATEGIES["swing_failure_delayed"](obj)["entry"]
check("coercing dtype does not change the signal",
      a.equals(b), f"  ({int(a.sum())} entries both ways)")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
