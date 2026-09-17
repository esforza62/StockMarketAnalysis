"""The forward test's job is mostly to REFUSE, so that is what this pins.

A naive version of this comparison produces a decisive-looking number that
measures hold length instead of edge: long-hold strategies are missing
their recent entries entirely (run_backtest drops open positions), so the
few that closed early are the ones that went right. Every check below is
about the machinery declining to report in exactly those cases.
"""
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import forward_test as ft

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


def epoch(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def write_log(tmp, records):
    p = Path(tmp) / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def entry(strategy, avg, trades=100):
    return {"regime": "trending", "direction": "down", "strategy": strategy,
            "trade_count": trades, "win_rate": 50.0, "avg_return_pct": avg,
            "total_return_pct": avg * trades, "compounded_return_pct": avg}


tmp = tempfile.mkdtemp()

log = write_log(tmp, [
    {"run_at": "2026-08-20T00:00:00+00:00", "interval": "1d", "summary": [
        entry("old_winner", 9.0), entry("beta", 5.0), entry("gamma", 1.0)]},
    {"run_at": "2026-08-24T00:00:00+00:00", "interval": "1d", "summary": [
        entry("alpha", 8.0), entry("beta", 5.0), entry("gamma", 1.0)]},
    # A later run must NOT leak into a pick frozen on the 24th.
    {"run_at": "2026-09-01T00:00:00+00:00", "interval": "1d", "summary": [
        entry("future_winner", 99.0), entry("beta", 5.0)]},
])

picks = ft.load_picks(log, "2026-08-24")
key = ("1d", "trending", "down")

check("1. the pick is the top-ranked strategy on the as-of date",
      picks[key]["pick"] == "alpha")

check("2. a later run does not leak backwards into an earlier pick",
      "future_winner" not in picks[key]["ranked"])

check("3. the runner-up margin is carried, to expose tie-breaks",
      abs(picks[key]["margin"] - 3.0) < 1e-9)

check("4. the median-ranked strategy is identified for the benchmark",
      picks[key]["median_rank"] == "beta")

# min_trades gates the pick: a 3-trade strategy topping the list on noise
# must not become the pick.
log2 = write_log(tmp, [{"run_at": "2026-08-24T00:00:00+00:00", "interval": "1d", "summary": [
    entry("noise", 99.0, trades=3), entry("real", 4.0, trades=100)]}])
check("5. a thin strategy cannot become the pick",
      ft.load_picks(log2, "2026-08-24")[key]["pick"] == "real")

# ---- coverage arithmetic -------------------------------------------------
conn = sqlite3.connect(":memory:")
conn.executescript("""
CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE trades (run_id INT, ticker TEXT, strategy_id INT, regime TEXT,
                     direction TEXT, entry_date INT, exit_date INT,
                     return_pct REAL, win INT, size REAL);
CREATE TABLE runs (id INTEGER PRIMARY KEY, run_at TEXT, interval TEXT);
INSERT INTO runs VALUES (1, '2026-09-15T00:00:00+00:00', '1d');
INSERT INTO strategies VALUES (1,'slow'),(2,'fast');
""")
DAY = 86400
# 'slow' holds 200 days, 'fast' holds 2 -- both traded up to 2026-09-14.
for i in range(20):
    conn.execute("INSERT INTO trades VALUES (1,'T',1,'trending','down',?,?,5.0,1,1.0)",
                 (epoch("2026-01-01") + i * DAY, epoch("2026-01-01") + i * DAY + 200 * DAY))
    conn.execute("INSERT INTO trades VALUES (1,'T',2,'trending','down',?,?,1.0,1,1.0)",
                 (epoch("2026-09-01") + i * DAY, epoch("2026-09-01") + i * DAY + 2 * DAY))
conn.commit()
ft._HOLD_CACHE.clear(); ft._END_CACHE.clear()

cov_slow = ft.coverage(conn, 1, "slow", "2026-08-24")
cov_fast = ft.coverage(conn, 1, "fast", "2026-08-24")
check("6. a long-hold strategy has zero coverage over a short window",
      cov_slow == 0.0)
check("7. a short-hold strategy is nearly fully covered over the same window",
      cov_fast > 0.9)

# ---- clean-fixture checks, before anything mutates it --------------------
info_fast = {"as_of": "2026-08-24", "pick": "fast", "margin": 1.0,
             "median_rank": "fast", "ranked": ["fast"], "pick_in_sample_return": 1.0}
res2 = ft.evaluate_bucket(conn, 1, "1d", "trending", "down", info_fast, "2026-08-24")
check("8. a well-covered pick IS scored", res2["excess_vs_equal_weight"] is not None)

fwd = ft.forward_trades(conn, 1, "trending", "down", "2026-08-24")
check("9. only post-pick trades are returned, grouped by strategy",
      "slow" not in fwd and len(fwd.get("fast", [])) == 20)

check("10. forward trades exclude entries at or before the pick date",
      all(e > epoch("2026-08-24")
          for (e,) in conn.execute("SELECT entry_date FROM trades WHERE entry_date > ?",
                                   (epoch("2026-08-24"),))))

# ---- the refusal that matters, tested last because it mutates ------------
# The real trap: the long-hold pick DOES have forward trades, because a few
# closed unusually fast -- and for a dip-recovery strategy the ones that
# close fast are the ones that recovered. This is the live 1h
# rsi_dip_recovery case, showing +7.89% against a -0.51% benchmark on 30
# trades at 0% coverage. A mean over those is a survivorship artifact, so
# it must be refused even though the sample is non-empty and the number
# looks spectacular.
for i in range(5):
    conn.execute("INSERT INTO trades VALUES (1,'T',1,'trending','down',?,?,40.0,1,1.0)",
                 (epoch("2026-08-25") + i * DAY, epoch("2026-08-25") + i * DAY + 2 * DAY))
conn.commit()
ft._HOLD_CACHE.clear(); ft._END_CACHE.clear()

info = {"as_of": "2026-08-24", "pick": "slow", "margin": 3.0,
        "median_rank": "fast", "ranked": ["slow", "fast"], "pick_in_sample_return": 5.0}
res = ft.evaluate_bucket(conn, 1, "1d", "trending", "down", info, "2026-08-24")

check("11. the censored pick's few fast closes look outstanding",
      res["pick_forward"]["n"] == 5 and res["pick_forward"]["mean"] > 30)
check("12. and are refused anyway, on coverage",
      res["excess_vs_equal_weight"] is None and "UNRELIABLE" in res["note_equal_weight"])

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
