"""Slippage is charged per fill, and the default must be inert.

Every figure on the desk was produced without it, so switching it on by
accident would silently restate published numbers. It also interacts with
stops in a way that is easy to miss: the stop level is derived from the
entry price, so slipping the entry moves the stop and can change WHICH
bars a stopped strategy trades, not merely what it earns on them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import _slipped, backtest_strategy, run_backtest
from smc_regime.strategies import STRATEGIES

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


def frame(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Open": closes, "High": [c * 1.02 for c in closes], "Low": [c * 0.98 for c in closes],
         "Close": closes, "Volume": [1000] * len(closes)}, index=idx)


def signals(entries, exits, index):
    return pd.DataFrame({"entry": entries, "exit": exits}, index=index)


def synth(seed, n=500):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0.0003, 0.02, n)))
    return pd.DataFrame(
        {"Open": c, "High": c * (1 + abs(r.normal(0, 0.008, n))),
         "Low": c * (1 - abs(r.normal(0, 0.008, n))), "Close": c,
         "Volume": r.integers(1e5, 1e6, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="D"))


# ---- the haircut itself ---------------------------------------------------
check("1. buying pays up", _slipped(100.0, True, 0.05) == 100.05)
check("2. selling receives less", _slipped(100.0, False, 0.05) == 99.95)
check("3. zero is an exact no-op", _slipped(100.0, True, 0.0) == 100.0
      and _slipped(100.0, False, 0.0) == 100.0)

# A round trip costs twice the per-side rate, which is the point: it makes
# turnover expensive rather than applying one flat haircut to the result.
df = frame([100.0, 100.0, 100.0, 100.0])
sig = signals([False, True, False, False], [False, False, True, False], df.index)
plain = run_backtest(df, sig)[0]
slipped = run_backtest(df, sig, slippage_pct=0.10)[0]
check("4. a flat market returns zero without slippage", abs(plain.return_pct) < 1e-9)
check("5. and a round trip costs about twice the per-side rate",
      abs(slipped.return_pct - (-0.2)) < 0.01)

# ---- every exit path wears it, including the stop ------------------------
falling = frame([100.0, 100.0, 90.0, 80.0])
sig = signals([False, True, False, False], [False] * 4, falling.index)
stopped = run_backtest(falling, sig, stop_loss_pct=5.0, slippage_pct=0.10)[0]
unstopped = run_backtest(falling, sig, stop_loss_pct=5.0)[0]
check("6. a stop fill is slipped -- a stop is a level you wanted, not a fill",
      stopped.exit_price < unstopped.exit_price)

held = run_backtest(falling, sig, max_hold_bars=1, slippage_pct=0.10)[0]
check("7. a max-hold exit is slipped too",
      held.exit_price < run_backtest(falling, sig, max_hold_bars=1)[0].exit_price)

rising = frame([100.0, 100.0, 110.0, 120.0])
sig = signals([False, True, False, False], [False] * 4, rising.index)
open_slip = run_backtest(rising, sig, slippage_pct=0.10)[0]
open_plain = run_backtest(rising, sig)[0]
check("8. the open position's mark wears it, so opens stay comparable with closes",
      open_slip.is_open and open_plain.is_open
      and open_slip.exit_price < open_plain.exit_price)

# ---- the default must not move anything ---------------------------------
identical = True
for seed in range(3):
    d = synth(seed)
    for s in sorted(STRATEGIES):
        a = [(t.entry_price, t.exit_price, t.is_open) for t in backtest_strategy(d, s)]
        b = [(t.entry_price, t.exit_price, t.is_open) for t in backtest_strategy(d, s, slippage_pct=0.0)]
        if a != b:
            identical = False
check("9. slippage_pct=0.0 is identical to omitting it, every strategy", identical)

# ---- and the interaction that is easy to miss ---------------------------
# A stopless strategy trades the same bars either way: only the prices move.
d = synth(1)
bars_plain = [(t.entry_date, t.exit_date) for t in backtest_strategy(d, "macd")]
bars_slip = [(t.entry_date, t.exit_date) for t in backtest_strategy(d, "macd", slippage_pct=0.10)]
check("10. a stopless strategy trades the same bars, only at worse prices",
      bars_plain == bars_slip)

# A STOPPED one does not: the stop is derived from the entry price, so
# slipping the entry moves the stop and it can trigger on a different bar.
# This is correct, and it means a slipped run of a stopped strategy is not
# comparable trade-for-trade with an unslipped one.
sf_plain = [(t.entry_date, t.exit_date) for t in backtest_strategy(d, "swing_failure_delayed")]
sf_slip = [(t.entry_date, t.exit_date) for t in backtest_strategy(d, "swing_failure_delayed", slippage_pct=0.10)]
check("11. a stopped strategy's trade set DOES shift, because the stop moves with the entry",
      sf_plain != sf_slip)

# Costs must be monotonic -- more slippage can never earn more.
d = synth(2)
totals = [np.sum([t.return_pct for t in backtest_strategy(d, "macd", slippage_pct=bp)
                  if not t.is_open]) for bp in (0.0, 0.01, 0.05, 0.10)]
check("12. a stopless strategy's return falls monotonically with cost",
      all(a >= b for a, b in zip(totals, totals[1:])))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
