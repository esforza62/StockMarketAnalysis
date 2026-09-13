"""Round-trip checks for smc_regime.grade_history."""
import json
from pathlib import Path
import tempfile

import pandas as pd

from smc_regime import grade_history as gh

tmp = Path(tempfile.mkdtemp())
path = tmp / "nested" / "hist.jsonl"

# 1. load() on a file that does not exist yet
empty = gh.load(path)
assert empty.empty, "expected empty frame"
expected_cols = ["as_of", "interval", *gh._BASE_FIELDS, *gh._COMPONENT_FIELDS, "valuation_available", *gh._NEWS_FIELDS]
assert list(empty.columns) == expected_cols, f"columns wrong: {list(empty.columns)}"
print("1. absent file -> empty frame with", len(empty.columns), "columns  OK")

def frame(rows):
    return pd.DataFrame(rows)

base = {
    "industry": "Software", "streak_bars": 4, "borderline": None,
    **{f: 5 for f in gh._COMPONENT_FIELDS},
}

# 2. to_records with close carried on the frame
scores = frame([
    {"ticker": "MSFT", "sector": "Technology", "grade": "A", "total_points": 82.34567,
     "close": 415.123456, "regime": "trending", "direction": "bullish", **base},
    {"ticker": "AAPL", "sector": "Technology", "grade": "B", "total_points": 71.0,
     "close": float("nan"), "regime": "ranging", "direction": "neutral", **base},
])
recs = gh.to_records(scores, "2026-09-01", "1d", closes={"AAPL": 220.5})
assert recs[0]["close"] == 415.1235, recs[0]["close"]
assert recs[0]["total_points"] == 82.3457, recs[0]["total_points"]
assert recs[1]["close"] == 220.5, recs[1]["close"]   # NaN on frame -> closes fallback
assert recs[0]["valuation_available"] is True
print("2. to_records: rounding, NaN->closes fallback  OK")

# 3. missing entirely in both -> None, and JSON-serialisable
scores_missing = frame([
    {"ticker": "ZZZZ", "sector": "Energy", "grade": "D", "total_points": 30.0,
     "close": float("nan"), "regime": "ranging", "direction": "neutral", **base},
])
r = gh.to_records(scores_missing, "2026-09-01", "1d", valuation_available=False)[0]
assert r["close"] is None, r["close"]
assert r["valuation_available"] is False
json.dumps(r)
print("3. no close anywhere -> None, still serialisable  OK")

# 4. append creates parent dirs and returns row count
n = gh.append(recs, path)
assert path.exists() and n == 2, n
print("4. append created", path.parent.name + "/", "and holds", n, "rows  OK")

# 5. dedupe: re-run the same date, last write wins
corrected = gh.to_records(frame([
    {"ticker": "MSFT", "sector": "Technology", "grade": "C", "total_points": 60.0,
     "close": 400.0, "regime": "ranging", "direction": "neutral", **base},
]), "2026-09-01", "1d")
n = gh.append(corrected, path)
assert n == 2, f"expected dedupe to 2 rows, got {n}"
df = gh.load(path)
assert df.loc[df.ticker == "MSFT", "grade"].item() == "C", "last write did not win"
print("5. re-running a date corrects rather than duplicates  OK")

# 6. same ticker+date on a different interval is a distinct row
n = gh.append(gh.to_records(scores, "2026-09-01", "1w"), path)
assert n == 4, n
print("6. interval is part of the key  OK")

# 7. sort order, and idempotent normalize
gh.append(gh.to_records(scores, "2026-08-25", "1d"), path)
before = path.read_text()
assert gh.normalize(path) == 6
assert path.read_text() == before, "normalize is not idempotent"
keys = [(r["as_of"], r["interval"], r["ticker"]) for r in
        (json.loads(l) for l in path.read_text().splitlines())]
assert keys == sorted(keys), keys
print("7. sorted by (as_of, interval, ticker), normalize idempotent  OK")

# 8. interval filter on load
d1 = gh.load(path, interval="1d")
assert set(d1["interval"]) == {"1d"} and len(d1) == 4, len(d1)
assert list(d1.index) == list(range(len(d1))), "index not reset"
print("8. load(interval=) filters and resets index  OK")


# 9. The sentiment reading rides along with the grade. It is stored as the
#    number and the sample size only -- never the band label, which is a
#    function of thresholds calibrated to one night's corpus and would
#    freeze that calibration into the history.
news_frame = frame([
    {"ticker": "AAPL", "sector": "Tech", "grade": "A", "total_points": 80.0, "close": 220.5,
     "regime": "trending", "direction": "up",
     "news": {"article_count": 20, "avg_compound": 0.123456, "label": "typical"}, **base},
    {"ticker": "MSFT", "sector": "Tech", "grade": "B", "total_points": 70.0, "close": 400.0,
     "regime": "trending", "direction": "up",
     "news": {"article_count": 0, "avg_compound": None, "label": "no data"}, **base},
])
n_rows = gh.to_records(news_frame, "2026-09-01", "1d")
assert n_rows[0]["news_compound"] == 0.1235, n_rows[0]["news_compound"]
assert n_rows[0]["news_articles"] == 20
assert n_rows[1]["news_compound"] is None and n_rows[1]["news_articles"] == 0
assert "news_label" not in n_rows[0] and "label" not in n_rows[0], "the band label must not be stored"
json.dumps(n_rows)  # must stay serialisable -- numpy types would not be
print("9. news_compound/news_articles captured, label deliberately not  OK")

# 10. The backfill carries no news column at all: historical headlines
#     cannot be reconstructed, so the fields must read empty rather than
#     inventing a neutral score that would look like a real reading.
no_news = gh.to_records(frame([
    {"ticker": "AAPL", "sector": "Tech", "grade": "A", "total_points": 80.0, "close": 220.5,
     "regime": "trending", "direction": "up", **base},
]), "2026-09-01", "1d")[0]
assert no_news["news_compound"] is None and no_news["news_articles"] is None
print("10. a frame with no news column yields empty fields, not zeros  OK")

# 11. Rows written before these fields existed still load with the full
#     column set -- otherwise a first run after the change returns a frame
#     one column short of every other path.
legacy = tmp / "legacy.jsonl"
legacy.write_text(json.dumps({
    "as_of": "2026-08-01", "interval": "1d", "ticker": "AAPL", "sector": "Tech",
    "grade": "A", "total_points": 80.0, "close": 220.5, "regime": "trending",
    "direction": "up", "valuation_available": True,
    **{f: 5 for f in gh._COMPONENT_FIELDS},
}) + "\n")
loaded = gh.load(legacy)
assert list(loaded.columns) == expected_cols, f"columns wrong: {list(loaded.columns)}"
assert pd.isna(loaded.loc[0, "news_compound"]), "an older row should read NaN, not a value"
print("11. rows predating the fields load with them as NaN  OK")

print("\nall checks passed")
