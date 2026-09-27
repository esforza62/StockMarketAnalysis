"""Grades must be filed under the session they describe.

They were filed under the run's wall-clock UTC date, on the reasoning that
the nightly fires at 21:00 UTC so the two coincide. GitHub's scheduler
drifted to 23:11-23:59, so the 1d step began landing either side of
midnight and four of six runs filed wrongly -- one capture collided with
another and was deduped away outright. The dataset whose entire value is
accumulating correctly over time was losing roughly a capture in three.
"""
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.capture_grades import _run_date

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


def epoch(day, hour=20):
    return int(datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00").timestamp())


def make(run_at, interval, last_bar, *, is_open=True, extra_bars=()):
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE runs (id INTEGER PRIMARY KEY, run_at TEXT, interval TEXT);
        CREATE TABLE trades (run_id INT, ticker TEXT, strategy_id INT, regime TEXT,
                             direction TEXT, entry_date INT, exit_date INT,
                             return_pct REAL, win INT, size REAL, is_open INT);
    """)
    conn.execute("INSERT INTO runs VALUES (1,?,?)", (run_at, interval))
    for day in extra_bars:
        conn.execute("INSERT INTO trades VALUES (1,'T',1,'trending','down',?,?,1.0,1,1.0,0)",
                     (epoch(day) - 86400 * 10, epoch(day)))
    conn.execute("INSERT INTO trades VALUES (1,'T',1,'trending','down',?,?,1.0,1,1.0,?)",
                 (epoch(last_bar) - 86400 * 10, epoch(last_bar), 1 if is_open else 0))
    conn.commit()
    return conn


# run 41: started 23:50, the 1d step finished 00:10 the NEXT day, so the
# wall clock said Friday while the data ended on Thursday.
c = make("2026-09-25T00:10:17+00:00", "1d", "2026-09-24")
check("1. a run that finished past midnight files under the session, not the clock",
      _run_date(c, "1d") == "2026-09-24")

# run 42: the same slip carried a Friday session onto a SATURDAY date.
c = make("2026-09-26T00:10:00+00:00", "1d", "2026-09-25")
check("2. a Friday session is never filed under a Saturday",
      _run_date(c, "1d") == "2026-09-25")

# run 40: finished at 23:45, before midnight. Must be unchanged.
c = make("2026-09-23T23:45:39+00:00", "1d", "2026-09-23")
check("3. a run that finished before midnight is unaffected",
      _run_date(c, "1d") == "2026-09-23")

# The collision: two runs a day apart must now produce DIFFERENT dates.
a = _run_date(make("2026-09-22T00:11:37+00:00", "1d", "2026-09-21"), "1d")
b = _run_date(make("2026-09-22T23:47:17+00:00", "1d", "2026-09-22"), "1d")
check("4. consecutive runs no longer collide on one date", a != b)
check("5. and each names its own session", (a, b) == ("2026-09-21", "2026-09-22"))

# Weekly bars are labelled by their week-ending Friday, so a weekly run
# legitimately reports a later date than the daily one from the same night.
c = make("2026-09-25T01:34:00+00:00", "1w", "2026-09-25")
check("6. a weekly run files under its week-ending bar", _run_date(c, "1w") == "2026-09-25")

# A database written before is_open existed has no open rows to read, and
# must fall back rather than return nothing.
c = make("2026-09-25T00:10:00+00:00", "1d", "2026-09-24", is_open=False)
check("7. with no open trades it falls back to the newest exit",
      _run_date(c, "1d") == "2026-09-24")

# The newest exit must win, not whichever row happens to be last inserted.
c = make("2026-09-25T00:10:00+00:00", "1d", "2026-09-24",
         extra_bars=("2026-09-10", "2026-09-17"))
check("8. older trades do not drag the date backwards",
      _run_date(c, "1d") == "2026-09-24")

# No trades at all: the old wall-clock behaviour is the last resort.
conn = sqlite3.connect(":memory:")
conn.executescript("""
    CREATE TABLE runs (id INTEGER PRIMARY KEY, run_at TEXT, interval TEXT);
    CREATE TABLE trades (run_id INT, ticker TEXT, strategy_id INT, regime TEXT,
                         direction TEXT, entry_date INT, exit_date INT,
                         return_pct REAL, win INT, size REAL, is_open INT);
    INSERT INTO runs VALUES (1,'2026-09-25T00:10:00+00:00','1d');
""")
conn.commit()
check("9. with no trades it falls back to the run timestamp",
      _run_date(conn, "1d") == "2026-09-25")

check("10. an unknown interval yields no date rather than a wrong one",
      _run_date(conn, "4h") is None)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
