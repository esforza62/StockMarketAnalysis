"""Ranked selection and the quality floor.

Two separate mechanisms, and conflating them is the mistake to guard
against. RANKING decides who gets a scarce slot; it changes WHICH signals
are taken but not how many. The FLOOR declines a signal even when a slot
stands free, so it changes HOW MUCH of the book is invested at all -- which
is the point, because a book that is always fully invested in 40 names from
a 415-name large-cap universe is an index fund with tracking error, whatever
rule chose the names.

So the checks below insist on both: that ranking actually orders by score,
and that a floor leaves capital in cash rather than quietly backfilling the
slot with the next candidate down.
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


# A, B, C all signal on bar 1 and exit on bar 2. Their PAYOFFS differ, so a
# selector that ignores the score lands on a different answer than one that
# honours it -- that is what makes these checks discriminating rather than
# merely descriptive.
pay = {"A": 200.0, "B": 150.0, "C": 110.0}
prices = {t: frame([100, 100, pay[t]] + [pay[t]] * 9) for t in "ABC"}
trades = {t: [Trade(IDX[1], IDX[2], 100.0, pay[t])] for t in "ABC"}
# score ordering A > B > C, deliberately matching payoff order so a correct
# ranked run is identifiable from the equity curve alone
scores = {("A", IDX[1]): 9.0, ("B", IDX[1]): 5.0, ("C", IDX[1]): 1.0}

# --- 1-3: ranking picks the best score, not the first symbol --------------
r1 = simulate(trades, prices, max_positions=1, capital=1000.0,
              selector="ranked", scores=scores)
check("1. ranked takes exactly one signal when one slot is free",
      r1.taken == 1 and r1.rejected == 2)
check("2. ranked took the HIGHEST-scored candidate",
      abs(r1.equity.iloc[-1] - 2000.0) < 1e-6,
      f"  (final equity {r1.equity.iloc[-1]:.2f}, A doubles 1000 -> 2000)")
check("3. a slot taken on quality is not recorded as declined",
      r1.declined == 0)

# Flip the scores: same trades, same prices, only the ordering reversed. If
# ranking works, the outcome must follow the score rather than the data.
flipped = {("A", IDX[1]): 1.0, ("B", IDX[1]): 5.0, ("C", IDX[1]): 9.0}
r2 = simulate(trades, prices, max_positions=1, capital=1000.0,
              selector="ranked", scores=flipped)
check("4. reversing the scores changes which trade is taken",
      abs(r2.equity.iloc[-1] - 1100.0) < 1e-6,
      f"  (final equity {r2.equity.iloc[-1]:.2f}, C rises 10% -> 1100)")

# --- 5-8: the floor holds CASH rather than backfilling the slot -----------
# Floor above every score: nothing qualifies, and with 3 slots free the book
# must stay in cash rather than take the best of a bad field.
r3 = simulate(trades, prices, max_positions=3, capital=1000.0,
              selector="ranked", scores=scores, score_floor=9.5)
check("5. a floor above every score takes nothing",
      r3.taken == 0 and r3.declined == 3)
check("6. declining on quality leaves capital in cash, not reallocated",
      abs(r3.equity.iloc[-1] - 1000.0) < 1e-6,
      f"  (final equity {r3.equity.iloc[-1]:.2f}, unchanged)")
check("7. a signal declined on quality is NOT counted as slot-rejected",
      r3.rejected == 0,
      "  (rejected means the book was full; declined means it chose cash)")
check("8. an all-cash book reports zero slot fill and zero capital deployed",
      r3.stats["slot_fill_pct"] == 0.0 and r3.stats["capital_deployed_pct"] == 0.0)

# Floor between B and C: A and B qualify, C does not, and 3 slots are free --
# so exactly two positions open and a third of the book stays cash.
r4 = simulate(trades, prices, max_positions=3, capital=900.0,
              selector="ranked", scores=scores, score_floor=5.0)
check("9. a mid floor takes only the qualifying signals",
      r4.taken == 2 and r4.declined == 1 and r4.rejected == 0)
# A: 300 -> 600, B: 300 -> 450, C declined: 300 stays cash. Total 1350.
check("10. the partially-filled book is valued correctly",
      abs(r4.equity.iloc[-1] - 1350.0) < 1e-6,
      f"  (final equity {r4.equity.iloc[-1]:.2f}, expected 1350 = 600+450+300 cash)")
check("11. slot fill stays below 100% when the floor declines a candidate",
      0.0 < r4.stats["slot_fill_pct"] < 100.0,
      f"  ({r4.stats['slot_fill_pct']:.1f}%)")

# --- 12-13: an unscored signal is unrankable, not implicitly good ---------
r5 = simulate(trades, prices, max_positions=3, capital=900.0, selector="ranked",
              scores={("A", IDX[1]): 9.0}, score_floor=1.0)
check("12. a signal with no score is declined under a floor, not waved through",
      r5.taken == 1 and r5.declined == 2,
      "  (only A is scored, so only A can clear a floor)")

# --- 13-15: misuse raises rather than silently passing everything --------
for label, kw in (("selector='ranked' without scores", {"selector": "ranked"}),
                  ("score_floor without scores", {"score_floor": 1.0})):
    try:
        simulate(trades, prices, max_positions=1, **kw)
        raised = False
    except ValueError:
        raised = True
    check(f"13. {label} raises", raised)

try:
    simulate(trades, prices, max_positions=1, selector="nonsense")
    raised = False
except ValueError:
    raised = True
check("14. an unknown selector still raises", raised)

# --- 15: ranking is deterministic, so it needs no seed averaging ----------
a = simulate(trades, prices, max_positions=1, capital=1000.0, selector="ranked",
             scores=scores, seed=0).equity.iloc[-1]
b = simulate(trades, prices, max_positions=1, capital=1000.0, selector="ranked",
             scores=scores, seed=99).equity.iloc[-1]
check("15. ranked ignores the seed -- one answer, no tie-break dispersion",
      a == b)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
