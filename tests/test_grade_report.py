"""Forward-return maths in smc_regime.grade_report, on a history with known answers."""
import tempfile
from pathlib import Path

import pandas as pd

from smc_regime import grade_history as gh
from smc_regime import grade_report as gr

tmp = Path(tempfile.mkdtemp())
path = tmp / "h.jsonl"

def row(as_of, ticker, grade, close, val=False):
    return {"as_of": as_of, "interval": "1d", "ticker": ticker, "sector": "S",
            "grade": grade, "total_points": 80.0, "close": close,
            "regime": "trending", "direction": "up",
            **{f: 5.0 for f in gh._COMPONENT_FIELDS}, "valuation_available": val}

# AAA doubles over 7 days, BBB halves. Weekly cadence so the 1w horizon
# lands exactly and 1m/3m stay pending.
recs = [
    row("2026-09-01", "AAA", "A", 100.0), row("2026-09-01", "BBB", "D", 100.0),
    row("2026-09-08", "AAA", "A", 200.0), row("2026-09-08", "BBB", "D", 50.0),
]
gh.append(recs, path)

scored = gr.forward_returns(gh.load(path))
a = scored[(scored.ticker == "AAA") & (scored.as_of == "2026-09-01")].iloc[0]
b = scored[(scored.ticker == "BBB") & (scored.as_of == "2026-09-01")].iloc[0]
assert round(a["fwd_1w"], 6) == 100.0, a["fwd_1w"]
assert round(b["fwd_1w"], 6) == -50.0, b["fwd_1w"]
print("1. forward return maths: +100% / -50% over 7 days  OK")

# The last date has no future row: 1w has elapsed for 09-01 but not 09-08.
last = scored[scored.as_of == "2026-09-08"]
assert last["fwd_1w"].isna().all(), "invented a return with no later close"
assert not last["mature_1w"].any(), "unelapsed horizon marked mature"
assert scored[scored.as_of == "2026-09-01"]["mature_1w"].all()
print("2. unelapsed horizon is pending, not zero  OK")

s = gr.summarize(scored)
h = s["horizons"]["1w"]
assert h["grades"]["A"] == {"n": 1, "mean_pct": 100.0, "median_pct": 100.0, "win_rate_pct": 100.0}, h["grades"]["A"]
assert h["grades"]["D"]["mean_pct"] == -50.0
assert h["spread_a_minus_d_pct"] == 150.0 and h["monotonic"] is True
assert h["pending_rows"] == 2 and h["gap_rows"] == 0
assert s["horizons"]["1m"]["grades"]["A"]["n"] == 0 and s["horizons"]["1m"]["pending_rows"] == 4
print(f"3. summarize: spread {h['spread_a_minus_d_pct']}%, monotonic, 2 pending  OK")

# A history hole beyond the slack window must not be stretched into a return.
gap = [row("2026-10-20", "AAA", "A", 300.0), row("2026-10-20", "BBB", "D", 10.0)]
gh.append(gap, path)
scored2 = gr.forward_returns(gh.load(path))
r = scored2[(scored2.ticker == "AAA") & (scored2.as_of == "2026-09-08")].iloc[0]
assert pd.isna(r["fwd_1w"]), f"stretched a 42-day gap into a 1w return: {r['fwd_1w']}"
assert r["mature_1w"], "should be mature (elapsed) but unmeasurable"
h2 = gr.summarize(scored2)["horizons"]["1w"]
assert h2["gap_rows"] == 2, h2["gap_rows"]
print("4. 42-day gap is not counted as a 1w return; reported as unmeasurable  OK")

# Broken ordering must be called out, not smoothed over.
p2 = tmp / "bad.jsonl"
gh.append([row("2026-09-01", "AAA", "A", 100.0), row("2026-09-01", "BBB", "D", 100.0),
           row("2026-09-08", "AAA", "A", 90.0),  row("2026-09-08", "BBB", "D", 130.0)], p2)
bad = gr.summarize(gr.forward_returns(gh.load(p2)))["horizons"]["1w"]
assert bad["monotonic"] is False and bad["spread_a_minus_d_pct"] == -40.0, bad
print(f"5. D beating A -> monotonic False, spread {bad['spread_a_minus_d_pct']}%  OK")

# Mixed live/backfilled provenance is flagged.
gh.append([row("2026-09-01", "CCC", "B", 100.0, val=True)], path)
assert gr.summarize(gr.forward_returns(gh.load(path)))["valuation_mix"]["mixed"] is True
print("6. mixed live/backfilled valuation provenance flagged  OK")

t = gr.by_date(scored, "1w")
assert list(t.columns) == ["as_of", "A", "B", "C", "D", "spread", "n"], list(t.columns)
assert len(t) == 1 and t.iloc[0]["spread"] == 150.0
print("7. by_date table shape and spread  OK")

print("\nall checks passed")
