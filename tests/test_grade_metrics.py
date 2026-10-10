"""The tracked grade-quality metric, and the two things it must not hide.

A health metric that is wrong is worse than none: it gets watched, trusted,
and then quietly stops meaning anything. The checks here target the two
ways this one could mislead -- a pooled figure concealing a decaying
signal, and a mean dragged positive by a single lucky date -- plus the
arithmetic underneath.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import grade_metrics as gm

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


# --- 1-4: the rank correlation itself -----------------------------------
check("1. a perfectly ranked pair scores +1",
      abs(gm.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9)
check("2. a perfectly inverted pair scores -1",
      abs(gm.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) + 1.0) < 1e-9)
# Monotonic but wildly non-linear: Spearman must see this as perfect where
# a Pearson correlation on raw values would not. This is why it is rank-based.
check("3. rank correlation is blind to a non-linear but monotonic map",
      abs(gm.spearman([1, 2, 3, 4, 5], [1, 4, 9, 400, 10000]) - 1.0) < 1e-9)
check("4. a constant column yields nan rather than a spurious number",
      np.isnan(gm.spearman([1, 2, 3], [7, 7, 7])))


# --- 5-9: per-date IC is what makes degradation visible ------------------
def frame(rows):
    """rows: (as_of, ticker, score, fwd) -> the shape forward_returns leaves."""
    df = pd.DataFrame(rows, columns=["as_of", "ticker", "total_points", "fwd_2w"])
    df["mature_2w"] = True
    df["grade"] = np.where(df.total_points >= 76, "A",
                   np.where(df.total_points >= 68, "B",
                   np.where(df.total_points >= 57, "C", "D")))
    return df


# Two dates. On the first the score ranks perfectly; on the second it is
# perfectly INVERTED. A pooled IC over both can look unremarkable while the
# per-date series shows one good day and one bad one -- which is the whole
# reason the metric is computed per date.
good = [("2026-01-01", f"T{i}", 50 + i, float(i)) for i in range(40)]
bad = [("2026-01-02", f"T{i}", 50 + i, float(-i)) for i in range(40)]
ic = gm.information_coefficient(frame(good + bad), "2w")
check("5. per-date IC separates a good date from a bad one",
      len(ic["per_date"]) == 2
      and ic["per_date"]["2026-01-01"] > 0.99
      and ic["per_date"]["2026-01-02"] < -0.99,
      f"  ({ic['per_date']})")
check("6. the mean of a +1 and a -1 date is about zero",
      abs(ic["ic_mean"]) < 1e-6)
check("7. hit rate reports half the dates positive",
      ic["ic_hit_rate"] == 50.0)

# A mean can be dragged positive by one extreme date while the signal was
# wrong more often than right. hit_rate is what catches that, and the two
# are reported together for exactly this case.
lucky = [("2026-02-01", f"T{i}", 50 + i, float(i) * 100) for i in range(40)]
unlucky = [(f"2026-02-0{d}", f"T{i}", 50 + i, float(-i) * 0.01)
           for d in (2, 3, 4) for i in range(40)]
ic2 = gm.information_coefficient(frame(lucky + unlucky), "2w")
check("8. hit rate exposes a mean carried by one date",
      ic2["ic_hit_rate"] == 25.0 and ic2["ic_mean"] < 0,
      f"  (mean {ic2['ic_mean']:+.3f}, positive on {ic2['ic_hit_rate']:.0f}% of dates)")

# Thin dates are skipped rather than averaged in as noise.
thin = [("2026-03-01", f"T{i}", 50 + i, float(i)) for i in range(5)]
check("9. a date below the row floor is skipped, not counted",
      gm.information_coefficient(frame(thin + good), "2w")["dates"] == 1)


# --- 10-12: economic magnitude alongside the rank statistic --------------
check("10. the A-D spread is reported next to the IC",
      ic["spread_ad"] is not None and "winrate_gap_ad" in ic)
# An IC of +1 says the ORDER was right; it says nothing about the size of
# the payoff. Both are kept so a correct-but-worthless signal is visible.
tiny = [("2026-04-01", f"T{i}", 50 + i, i * 1e-9) for i in range(40)]
t = gm.information_coefficient(frame(tiny), "2w")
check("11. a perfectly ranked but economically tiny signal shows IC ~1 and spread ~0",
      t["per_date"]["2026-04-01"] > 0.99 and abs(t["spread_ad"]) < 1e-3,
      f"  (IC {t['per_date']['2026-04-01']:.2f}, spread {t['spread_ad']:+.5f})")
check("12. bucket sizes are reported so a thin bucket is visible",
      set(t["bucket_n"]) == {"A", "B", "C", "D"})


# --- 13-15: recording -----------------------------------------------------
tmp = Path(tempfile.mkdtemp()) / "m.jsonl"
check("13. an unwritten metrics file loads as empty, not an error",
      gm.load_recorded(tmp) == [])
m = {"computed_at": "2026-01-01T00:00:00Z", "history_rows": 10, "capture_dates": 2,
     "distribution": {}, "horizons": {"2w": dict(ic)}}
gm.record(m, tmp)
back = gm.load_recorded(tmp)
check("14. a recorded row round-trips", len(back) == 1 and back[0]["history_rows"] == 10)
# per_date would re-record every past date on every run, growing the file
# quadratically; it is recomputable from the history, so it is not stored.
check("15. the per-date map is dropped from the stored row",
      "per_date" not in back[0]["horizons"]["2w"]
      and back[0]["horizons"]["2w"]["ic_mean"] == ic["ic_mean"],
      "  (summary kept, the expanding map dropped)")
check("16. recording does not mutate the caller's dict",
      "per_date" in m["horizons"]["2w"])

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
