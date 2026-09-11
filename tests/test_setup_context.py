"""News and earnings ride along on a scored row without touching the grade.

The whole claim behind adding them is that they are REPORTED, not scored --
so the test that matters is that two identical setups, one with headlines
and an earnings date and one without, come out with the same number of
points and the same letter. If that ever stops being true, something has
quietly started weighting them.

Plain asserts, no pytest, no network -- see tests/README.md.
"""
import sys

sys.path.insert(0, "/home/user/StockMarketAnalysis")

from smc_regime.setup_score import score_ticker

BASE = dict(
    ticker="TEST",
    sector="Energy",
    industry="Oil & Gas",
    regime="trending",
    direction="up",
    streak_bars=12,
    snapshot={
        "close": 100.0, "ma_fast": 95.0, "ma_slow": 90.0,
        "rsi": 55.0, "macd_hist": 0.4, "macd_hist_prev": 0.2,
        "volume_ratio": 1.3, "price_change_pct": 1.1,
        "return_1w": 2.0, "return_1m": 5.0,
    },
    valuation=(20.0, 16.0),
)

NEWS = {"article_count": 7, "avg_compound": -0.61, "label": "very negative", "window_days": 7, "fetched_at": "2026-09-10T21:00:00+00:00"}

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


bare = score_ticker(**BASE)
loaded = score_ticker(**BASE, news=NEWS, earnings=("2026-10-29", False))

check(
    "1. news and earnings change neither the points nor the grade",
    bare["total_points"] == loaded["total_points"] and bare["grade"] == loaded["grade"],
)

# Every scored field, not just the total: a component could move while the
# total happened to land in the same place.
scored_fields = [k for k in bare if k.endswith("_points")]
check(
    "2. no component's points move either",
    bool(scored_fields) and all(bare[k] == loaded[k] for k in scored_fields),
)

check(
    "3. the context is carried onto the row",
    loaded["news"] == NEWS
    and loaded["earnings_date"] == "2026-10-29"
    and loaded["earnings_is_estimate"] is False,
)

check(
    "4. absent context is None, never a stand-in value",
    bare["news"] is None and bare["earnings_date"] is None and bare["earnings_is_estimate"] is None,
)

# A ticker Yahoo has valuation for but no earnings date -- the caller passes
# the pair through either way, so (None, None) must not become a date.
check(
    "5. a (None, None) earnings pair stays empty",
    score_ticker(**BASE, earnings=(None, None))["earnings_date"] is None,
)

# An estimated date has to stay distinguishable from a confirmed one all the
# way to the dashboard: it is the difference between a fixture and a guess
# that can move a week.
check(
    "6. the estimate flag survives as a real boolean",
    score_ticker(**BASE, earnings=("2026-10-30", True))["earnings_is_estimate"] is True,
)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
