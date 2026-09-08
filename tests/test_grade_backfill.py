"""smc_regime.grade_backfill on synthetic bars -- no Tiingo key needed."""
import json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from smc_regime import grade_backfill as gb
from smc_regime import grade_history as gh

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
META = pd.DataFrame({
    "ticker": TICKERS,
    "sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
    "industry": ["Software", "Software", "Semis", "Oil", "Oil", "Oil"],
})

def series(seed, drift, n=900):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.012, "Low": close * 0.988,
         "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)},
        index=idx,
    )

BARS = {t: series(i, d) for i, (t, d) in enumerate(
    zip(TICKERS, [0.0018, 0.0012, -0.0015, 0.0004, -0.0009, 0.0001]))}

def run(start, end, bars, path, **kw):
    gb.fetch_ohlcv = lambda ticker, interval="1d", start_date=None: bars[ticker].copy()
    gb.universe = lambda db_path: META.copy()
    return gb.backfill(start=start, end=end, db_path="unused", history_path=path,
                       cache_dir=None, **kw)

tmp = Path(tempfile.mkdtemp())

# --- 1. end-to-end over a date range -------------------------------------
p1 = tmp / "h1.jsonl"
dates = BARS["AAA"].index
start, end = dates[-6].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
graded, rows = run(start, end, BARS, p1)
assert graded == 6, graded
assert rows == 36, rows
df = gh.load(p1)
assert set(df["grade"]) <= set("ABCD"), set(df["grade"])
assert (~df["valuation_available"]).all(), "backfilled rows must be flagged"
assert df["close"].notna().all() and df["total_points"].between(0, 100).all()
assert (df["valuation_points"] == 5.0).all(), "valuation must sit at neutral half credit"
print(f"1. graded {graded} dates x 6 tickers -> {rows} rows; grades {dict(df.grade.value_counts())}  OK")

# --- 2. no lookahead -----------------------------------------------------
# Grading date D must not change when bars after D are removed.
cut = dates[-4]
truncated = {t: b.loc[:cut] for t, b in BARS.items()}
p2 = tmp / "h2.jsonl"
run(cut.strftime("%Y-%m-%d"), cut.strftime("%Y-%m-%d"), truncated, p2)
full_rows = {r["ticker"]: r for r in
             (json.loads(l) for l in p1.read_text().splitlines())
             if r["as_of"] == cut.strftime("%Y-%m-%d")}
trunc_rows = {r["ticker"]: r for r in
              (json.loads(l) for l in p2.read_text().splitlines())}
assert full_rows.keys() == trunc_rows.keys()
diffs = [t for t in full_rows if full_rows[t] != trunc_rows[t]]
assert not diffs, f"lookahead: {[(t, full_rows[t], trunc_rows[t]) for t in diffs]}"
print(f"2. {len(full_rows)} rows byte-identical with future bars removed -> no lookahead  OK")

# --- 3. peer counts are per-date and self-excluding -----------------------
g = gb._gather(META, BARS, {t: gb.confirmed_regime(gb.classify_regime(b)) for t, b in BARS.items()},
               dates[-1], "1d", 3, 200)
sec, ind = gb._peer_counts(g)
assert sum(sec["Tech"].values()) == 3 and sum(ind["Oil"].values()) == 3, (sec, ind)
assert set(sec) == {"Tech", "Energy"}, set(sec)
print(f"3. peer counts per date: sector {sec}  OK")

# --- 4. a ticker with no bar on the date is skipped, not carried forward --
halted = {t: (b.drop(index=dates[-1]) if t == "CCC" else b) for t, b in BARS.items()}
p4 = tmp / "h4.jsonl"
run(dates[-1].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"), halted, p4)
d4 = gh.load(p4)
assert "CCC" not in set(d4["ticker"]), "ticker with no bar was graded anyway"
assert len(d4) == 5, len(d4)
print("4. ticker that did not trade on the date is skipped  OK")

# --- 5. short history is skipped rather than scored on thin indicators ----
p5 = tmp / "h5.jsonl"
short = {t: (b.tail(120) if t == "DDD" else b) for t, b in BARS.items()}
run(dates[-1].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"), short, p5)
assert "DDD" not in set(gh.load(p5)["ticker"]), "under min_bars was graded"
print("5. ticker under --min-bars is skipped  OK")

# --- 6. rerunning a date corrects it in place -----------------------------
before = len(gh.load(p1))
run(start, end, BARS, p1)
assert len(gh.load(p1)) == before, "rerun duplicated rows"
print("6. rerunning the same range is idempotent  OK")

# --- 7. 1w interval works and is stored separately ------------------------
p7 = tmp / "h7.jsonl"
run(start, end, BARS, p7, interval="1w")
d7 = gh.load(p7)
assert set(d7["interval"]) == {"1w"} and len(d7) == 36, len(d7)
print("7. --interval 1w scores and stores separately  OK")

print("\nall checks passed")
