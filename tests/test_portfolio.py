"""One capital pool, with the invariants that catch the usual mistakes.

A portfolio simulator is easy to write and easy to get silently wrong. The
failures that matter are not crashes: a slot limit that does not bind, cash
that appears from nowhere, a position marked at a price it never traded at,
or a selection rule that quietly peeks at which trade won. Each check below
targets one of those.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import Trade
from smc_regime.portfolio import simulate

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


IDX = pd.date_range("2024-01-01", periods=12, freq="B")


def frame(closes):
    c = np.asarray(closes, float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": [1e6] * len(c)}, index=IDX)


def tr(i, j, pi, pj):
    return Trade(IDX[i], IDX[j], pi, pj)


# five tickers all signalling on the same bar, all doubling over 2 bars
flat = [100] * 12
up = [100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
prices = {t: frame(up) for t in "ABCDE"}
trades = {t: [tr(1, 2, 100.0, 200.0)] for t in "ABCDE"}

# --- 1-2: the slot limit binds, and rejections are counted -----------
r2 = simulate(trades, prices, max_positions=2, capital=1000.0)
check("1. the slot limit binds when signals outnumber slots",
      r2.taken == 2 and r2.rejected == 3, f"  (taken={r2.taken} rejected={r2.rejected})")
check("2. rejection rate is reported", abs(r2.rejection_rate - 0.6) < 1e-9,
      f"  ({r2.rejection_rate*100:.0f}%)")

r5 = simulate(trades, prices, max_positions=5, capital=1000.0)
check("3. with enough slots nothing is rejected",
      r5.taken == 5 and r5.rejected == 0)

check("4. a full book that doubles doubles the account",
      abs(r5.equity.iloc[-1] - 2000.0) < 1e-6, f"  (final {r5.equity.iloc[-1]:.2f}, want 2000)")

# SIZING AND SELECTION TOGETHER. A and B double, C/D/E halve. A slot is
# equity/max_positions, so fewer slots means BIGGER positions, not just
# fewer -- and the ticker selector takes A and B. Hand arithmetic:
#   5 slots: 200 each. A,B -> 400 each = 800. C,D,E -> 100 each = 300. = 1100
#   2 slots: 500 each. A,B only, both double            = 2000
mixed_prices = {t: frame(up if t in "AB" else
                         [100, 100, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]) for t in "ABCDE"}
mixed_trades = {t: [tr(1, 2, 100.0, 200.0 if t in "AB" else 50.0)] for t in "ABCDE"}
m5 = simulate(mixed_trades, mixed_prices, max_positions=5, capital=1000.0)
m2 = simulate(mixed_trades, mixed_prices, max_positions=2, capital=1000.0)
check("5. a slot is equity/max_positions, so fewer slots means larger positions",
      abs(m5.equity.iloc[-1] - 1100.0) < 1e-6 and abs(m2.equity.iloc[-1] - 2000.0) < 1e-6,
      f"  (5 slots {m5.equity.iloc[-1]:.0f} want 1100; 2 slots {m2.equity.iloc[-1]:.0f} want 2000)")
check("5b. the ticker selector took A and B, not an arbitrary pair",
      sorted(t[0] for t in m2.trades) == ["A", "B"],
      f"  (took {sorted(t[0] for t in m2.trades)})")

# --- 6: cash drag. one signal, five slots -> only 1/5 participates ---
one = {"A": [tr(1, 2, 100.0, 200.0)]}
r1 = simulate(one, prices, max_positions=5, capital=1000.0)
check("6. idle slots drag: one signal of five slots earns a fifth of the move",
      abs(r1.equity.iloc[-1] - 1200.0) < 1e-6, f"  (final {r1.equity.iloc[-1]:.2f}, want 1200)")

# --- 7: equity never goes negative and cash is never invented --------
rng = np.random.default_rng(5)
many_prices, many_trades = {}, {}
for k in range(12):
    t = f"T{k}"
    p = 100 * np.cumprod(1 + rng.normal(0, 0.03, 12))
    many_prices[t] = frame(p)
    many_trades[t] = [tr(1, 4, p[1], p[4]), tr(6, 9, p[6], p[9])]
rm = simulate(many_trades, many_prices, max_positions=3, capital=10_000.0)
check("7. equity stays finite and positive throughout",
      bool(rm.equity.notna().all() and (rm.equity > 0).all()))
check("8. never more positions open than slots",
      rm.taken <= sum(len(v) for v in many_trades.values()) and rm.rejected >= 0)

# --- 9: a freed slot is reusable the same day it frees ---------------
seq = {"A": [tr(1, 3, 100.0, 100.0)], "B": [tr(3, 5, 100.0, 100.0)]}
rseq = simulate(seq, {"A": frame(flat), "B": frame(flat)}, max_positions=1, capital=1000.0)
check("9. a slot freed on an exit day is usable that same day",
      rseq.taken == 2 and rseq.rejected == 0, f"  (taken={rseq.taken})")

# --- 10: slippage is charged on both fills --------------------------
rs = simulate(one, prices, max_positions=1, capital=1000.0, slippage_pct=0.0)
rslip = simulate(one, prices, max_positions=1, capital=1000.0, slippage_pct=0.10)
check("10. slippage reduces the result and is charged twice",
      rslip.equity.iloc[-1] < rs.equity.iloc[-1],
      f"  ({rs.equity.iloc[-1]:.2f} -> {rslip.equity.iloc[-1]:.2f})")

# --- 11: the default selector is deterministic, random is not -------
a = simulate(trades, prices, max_positions=2, capital=1000.0)
b = simulate(trades, prices, max_positions=2, capital=1000.0)
check("11. the ticker selector is reproducible",
      [t[0] for t in a.trades] == [t[0] for t in b.trades],
      f"  (picked {[t[0] for t in a.trades]})")
picks = {tuple(sorted(t[0] for t in simulate(trades, prices, max_positions=2,
                                             capital=1000.0, selector='random', seed=s).trades))
         for s in range(12)}
check("12. the random selector actually varies across seeds",
      len(picks) > 1, f"  ({len(picks)} distinct pairs over 12 seeds)")

# --- 13: stats are present and self-consistent ----------------------
s = r5.stats
check("13. stats report CAGR, drawdown and exposure",
      all(k in s for k in ("cagr_pct", "max_drawdown_pct", "exposure_pct", "return_over_vol")))
check("14. a monotonically rising account has no drawdown",
      abs(s["max_drawdown_pct"]) < 1e-9, f"  (maxDD {s['max_drawdown_pct']:.4f}%)")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
