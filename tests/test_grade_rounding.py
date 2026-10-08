"""The grade, the score and the borderline flag must agree with each other.

They did not when rounding carried a total onto a cut. The grade was struck
from the raw total while total_points reported round(total, 1), so a raw
score of 75.96 was graded B and reported as 76.0 -- and the documented cut
is A >= 76, so the row contradicted itself on its face. _borderline, reading
that same rounded 76.0, then reported the row as sitting "above" the cut with
B on the far side: the grade it already held. RJF did exactly this on the
2026-10-07 run, and 16 rows in the committed capture history carry it.

Checks 1-3 pin the helpers. Checks 4-6 drive the real score_ticker() path
over randomized inputs and hunt for raw totals that land in a rounding
window, so the wiring is tested rather than a restatement of the helpers.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.setup_score import (
    _BORDERLINE_MARGIN,
    _GRADE_THRESHOLDS,
    _borderline,
    _grade,
    score_ticker,
)

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


cuts = [t for t, _ in _GRADE_THRESHOLDS if t > 0]

# --- 1-3: the helpers -------------------------------------------------
ok = True
for cut in cuts:
    for raw in (cut - 0.04, cut - 0.01, cut - 0.001):
        if round(raw, 1) == float(cut) and _grade(round(raw, 1)) != _grade(float(cut)):
            ok = False
check("1. a total that rounds onto a cut grades as the cut", ok)

ok = True
x = 0.0
while x <= 100.0001:
    v = round(x, 2)
    b = _borderline(v)
    if b:
        here = _grade(v)
        across = _grade(b["cut"]) if v < b["cut"] else _grade(b["cut"] - 0.01)
        if b["adjacent"] != across or b["adjacent"] == here:
            ok = False
            break
    x += 0.01
check("2. borderline agrees with the thresholds across the whole range", ok)

ok = True
for cut in cuts:
    if _borderline(float(cut) + _BORDERLINE_MARGIN + 0.5) is not None:
        ok = False
    if _borderline(float(cut) + _BORDERLINE_MARGIN - 0.05) is None:
        ok = False
check("3. the borderline margin still bounds the flag", ok)


# --- 4-6: the real scoring path ---------------------------------------
def a_row(rng):
    """One scored row from randomized but realistic inputs."""
    close = rng.uniform(5, 500)
    return score_ticker(
        ticker="TEST",
        sector="Tech",
        industry="Semis",
        regime=rng.choice(["trending", "choppy"]),
        direction=rng.choice(["up", "down", "flat"]),
        streak_bars=rng.randint(1, 40),
        snapshot={
            "close": close,
            "ma_fast": close * rng.uniform(0.85, 1.15),
            "ma_slow": close * rng.uniform(0.85, 1.15),
            "rsi": rng.uniform(5, 95),
            "macd_hist": rng.uniform(-3, 3),
            "macd_hist_prev": rng.uniform(-3, 3),
            "volume_ratio": rng.uniform(0.3, 3.0),
            "price_change_pct": rng.uniform(-20, 20),
        },
        daily_row={"regime": rng.choice(["trending", "choppy"]), "direction": rng.choice(["up", "down", "flat"])},
        weekly_row={"regime": rng.choice(["trending", "choppy"]), "direction": rng.choice(["up", "down", "flat"])},
        sector_counts={"Tech": {"up": rng.randint(0, 9), "down": rng.randint(0, 9)}},
        industry_counts={"Semis": {"up": rng.randint(0, 9), "down": rng.randint(0, 9)}},
        valuation=(rng.uniform(1, 300), rng.uniform(1, 300)),
    )


rng = random.Random(20261007)
rows = [a_row(rng) for _ in range(4000)]

mismatched = [r for r in rows if _grade(r["total_points"]) != r["grade"]]
check("4. every row's grade matches its own reported total_points",
      not mismatched, f"  ({len(rows)} rows)")

selfcontra = [r for r in rows if r["borderline"] and r["borderline"]["adjacent"] == r["grade"]]
check("5. no row's borderline names that row's own grade", not selfcontra)

# The regression only bites when a raw total lands just under a cut and
# rounds onto it. Confirm the suite actually reached that window, or it is
# proving nothing about the thing that broke.
on_cut = [r for r in rows if float(r["total_points"]) in {float(c) for c in cuts}]
check("6. the sweep reached totals sitting exactly on a cut",
      bool(on_cut), f"  ({len(on_cut)} such rows)")
for r in on_cut:
    if _grade(r["total_points"]) != r["grade"]:
        failures.append("6b. a row on a cut graded against the unrounded total")
        break

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
