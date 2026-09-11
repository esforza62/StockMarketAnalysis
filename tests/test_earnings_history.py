"""The earnings history has one job: prove a date was known BEFORE the event.

Everything downstream depends on that. A filter tested against dates
reconstructed after the fact proves nothing about what was tradeable, so the
checks here are about the transition logic that turns "next report" into
"reported on", and about not recording claims the data never made.

Plain asserts, no pytest, no network -- see tests/README.md.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/user/StockMarketAnalysis")

import pandas as pd

from smc_regime import earnings_history as eh

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


tmp = Path(tempfile.mkdtemp()) / "earnings.jsonl"

# Night 1: three tickers, one with no date at all.
rows = eh.changed_rows(
    {"AAPL": {"earnings_date": "2026-10-29", "earnings_is_estimate": False},
     "CVX": {"earnings_date": "2026-10-30", "earnings_is_estimate": True},
     "SPY": {"earnings_date": None, "earnings_is_estimate": None}},
    {}, "2026-09-11")
eh.append(rows, tmp)
check("1. a ticker with no scheduled date is skipped, not recorded as null",
      len(rows) == 2 and all(r["ticker"] != "SPY" for r in rows))

# Night 2: nothing moved.
again = eh.changed_rows(
    {"AAPL": {"earnings_date": "2026-10-29", "earnings_is_estimate": False},
     "CVX": {"earnings_date": "2026-10-30", "earnings_is_estimate": True}},
    eh.latest(eh.load(tmp)), "2026-09-12")
check("2. an unchanged date writes nothing", len(again) == 0)

# Night 3: CVX's estimate firms up to a confirmed date -- same day, new claim.
firm = eh.changed_rows({"CVX": {"earnings_date": "2026-10-30", "earnings_is_estimate": False}},
                       eh.latest(eh.load(tmp)), "2026-10-01")
check("3. an estimate becoming confirmed is a change worth recording", len(firm) == 1)
eh.append(firm, tmp)

# Night 4: both report; the next dates roll forward.
eh.append(eh.changed_rows(
    {"AAPL": {"earnings_date": "2027-01-28", "earnings_is_estimate": True},
     "CVX": {"earnings_date": "2027-01-30", "earnings_is_estimate": True}},
    eh.latest(eh.load(tmp)), "2026-11-01"), tmp)

done = eh.reported_dates(eh.load(tmp))
aapl = done[done.ticker == "AAPL"]
check("4. a date rolling FORWARD yields the report date that just passed",
      len(aapl) == 1 and aapl.iloc[0]["reported_on"] == "2026-10-29")
check("5. the date is stamped as known before it happened",
      aapl.iloc[0]["known_by"] < aapl.iloc[0]["reported_on"])
check("6. a confirmed date is flagged apart from one still estimated",
      bool(aapl.iloc[0]["confirmed"]) is True)

# A correction BACKWARD is not a report.
eh.append(eh.changed_rows({"CVX": {"earnings_date": "2027-01-20", "earnings_is_estimate": True}},
                          eh.latest(eh.load(tmp)), "2026-12-01"), tmp)
cvx = eh.reported_dates(eh.load(tmp))
cvx = cvx[cvx.ticker == "CVX"]
check("7. a date moving EARLIER is a correction, not a report",
      "2027-01-30" not in set(cvx["reported_on"]))

# Idempotence and the conflict guard, same contract as grade_history.
before = eh.normalize(tmp)
check("8. normalize is idempotent", eh.normalize(tmp) == before)

with tmp.open("a") as f:
    f.write("<<<<<<< HEAD\n")
try:
    eh.normalize(tmp)
    check("9. a conflict marker raises rather than corrupting the file", False)
except ValueError:
    check("9. a conflict marker raises rather than corrupting the file", True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
