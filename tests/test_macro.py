"""The macro strip's failure modes are all silent ones.

An expired event calendar renders an empty schedule, which looks exactly
like a genuinely quiet fortnight. A level whose prior close was missing
renders a flat 0.00%, which looks exactly like a market that did not move.
Neither raises, and both are wrong in the direction of false reassurance,
so they are what these checks are about.
"""

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import macro

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


def cal(through, events):
    return {"covers_through": through, "events": events}


def ev(day, category, title, hour=13):
    at = (NOW + timedelta(days=day)).replace(hour=hour, minute=30)
    return {"at": at.isoformat().replace("+00:00", "Z"), "category": category, "title": title}


check(
    "1. a calendar covering today is not expired",
    not macro.calendar_expired(cal("2026-09-30", []), NOW),
)

check(
    "2. a calendar whose last date has passed IS expired",
    macro.calendar_expired(cal("2026-09-12", []), NOW),
)

# The dangerous case: the file is present and well-formed, just old. It
# must not read as "no events scheduled".
check(
    "3. an expired calendar is flagged even when it still lists events",
    macro.calendar_expired(cal("2026-09-01", [ev(-30, "fed", "Old Fed")]), NOW),
)

check(
    "4. a missing covers_through counts as expired, not as unlimited",
    macro.calendar_expired(cal(None, []), NOW) and macro.calendar_expired({}, NOW),
)

# Four CPI lines on one morning are one event to a reader.
same_day = cal("2026-12-31", [
    ev(3, "inflation", "Inflation Rate YoY"),
    ev(3, "inflation", "Inflation Rate MoM"),
    ev(3, "inflation", "Core Inflation Rate YoY"),
    ev(3, "fed", "Fed Interest Rate Decision"),
])
collapsed = macro.upcoming(same_day, NOW, days=14)
check(
    "5. same-day same-category events collapse to one row",
    len(collapsed) == 2 and {e["category"] for e in collapsed} == {"inflation", "fed"},
)

# 'other' is a grab-bag of unrelated releases, so collapsing it would hide
# real events behind each other.
others = cal("2026-12-31", [ev(2, "other", "Housing Starts"), ev(2, "other", "Building Permits")])
check(
    "6. unrelated same-day events are NOT collapsed together",
    len(macro.upcoming(others, NOW, days=14)) == 2,
)

horizon = cal("2026-12-31", [ev(2, "fed", "Soon"), ev(40, "fed", "Far")])
check(
    "7. events past the horizon are dropped",
    [e["title"] for e in macro.upcoming(horizon, NOW, days=14)] == ["Soon"],
)

# A release at 08:30 ET is still today's news at 18:00 UTC; dropping it the
# instant it prints would blank the most relevant chip on the page.
today_early = cal("2026-12-31", [ev(0, "inflation", "CPI", hour=6)])
check(
    "8. an event earlier today is still shown, not dropped as past",
    len(macro.upcoming(today_early, NOW, days=14)) == 1,
)

check(
    "9. events come back soonest first",
    [e["title"] for e in macro.upcoming(
        cal("2026-12-31", [ev(5, "fed", "Later"), ev(1, "jobs", "Sooner")]), NOW, days=14)]
    == ["Sooner", "Later"],
)

# A checkout without the file must render an honest empty, not raise.
missing = macro.load_calendar(Path(tempfile.mkdtemp()) / "nope.json")
check(
    "10. an absent calendar file loads as empty and expired",
    missing["events"] == [] and macro.calendar_expired(missing, NOW),
)

# The shipped file has to actually parse and carry the fields the page reads.
real = macro.load_calendar()
check(
    "11. the committed calendar parses and is well-formed",
    bool(real.get("events")) and bool(real.get("covers_through"))
    and all({"at", "title", "category"} <= set(e) for e in real["events"]),
)

check(
    "12. every committed event date is parseable and ordered-safe",
    all(datetime.fromisoformat(e["at"].replace("Z", "+00:00")) for e in real["events"]),
)

# ---- level parsing -----------------------------------------------------
# Both of these produce a plausible-looking wrong number rather than an
# error, which is the only reason they are worth pinning down.

import types


def fake_chart(bars, price, tz="America/New_York"):
    """A chart response with (epoch, close) bars and a live price."""
    return {
        "chart": {
            "error": None,
            "result": [{
                "meta": {"regularMarketPrice": price, "exchangeTimezoneName": tz,
                         "chartPreviousClose": 999.0},
                "timestamp": [b[0] for b in bars],
                "indicators": {"quote": [{"close": [b[1] for b in bars]}]},
            }],
        }
    }


def with_response(payload):
    class R:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return payload
    return types.SimpleNamespace(get=lambda *a, **k: R())


def epoch(y, m, d, hour=20):
    return int(datetime(y, m, d, hour, tzinfo=timezone.utc).timestamp())


real_requests = macro.requests

# Market shut: the latest bar already carries the current price, so the
# prior close must be the bar BEFORE it. Anchoring on "today" instead
# returns that same bar and every instrument reads a flat 0.00%.
try:
    macro.requests = with_response(fake_chart(
        [(epoch(2026, 9, 10), 100.0), (epoch(2026, 9, 11), 110.0)], price=110.0))
    shut = macro.fetch_level("TEST")
finally:
    macro.requests = real_requests

check(
    "13. with the market shut, change is measured against the previous session",
    shut["change_pct"] is not None and abs(shut["change_pct"] - 10.0) < 1e-9,
)

# Mid-session: today's partial bar is present and carries the live price.
# The prior close is yesterday, not the day before.
today = datetime.now(timezone.utc).date()
try:
    macro.requests = with_response(fake_chart([
        (epoch(2026, 9, 9), 50.0),
        (int(datetime(today.year, today.month, today.day, 12, tzinfo=timezone.utc).timestamp()) - 86400, 100.0),
        (int(datetime(today.year, today.month, today.day, 12, tzinfo=timezone.utc).timestamp()), 105.0),
    ], price=105.0))
    live = macro.fetch_level("TEST")
finally:
    macro.requests = real_requests

check(
    "14. mid-session, change is measured against yesterday, not two days back",
    live["change_pct"] is not None and abs(live["change_pct"] - 5.0) < 1e-9,
)

check(
    "15. an instrument with a bar dated today reports traded_today",
    live["traded_today"] is True and shut["traded_today"] is False,
)

# A single bar has nothing to compare against: None, never a flat zero.
try:
    macro.requests = with_response(fake_chart([(epoch(2026, 9, 11), 100.0)], price=100.0))
    lone = macro.fetch_level("TEST")
finally:
    macro.requests = real_requests

check(
    "16. one bar yields an unknown change rather than a flat zero",
    lone["change_pct"] is None,
)

check(
    "17. the equity benchmarks are configured as cash/future pairs",
    len(macro.INDEX_PAIRS) == 2
    and all(len(pair) == 2 and len(pair[0]) == 2 for pair in macro.INDEX_PAIRS),
)

# --- Futures rolls and settlement boundaries -------------------------------
#
# Both of these were live bugs on 2026-09-15, and both produced a number
# that looked completely ordinary on the page.


def rolled_chart(bars, price, vendor_change):
    """A chart whose bar series disagrees with the vendor's own change.

    That is exactly what a contract roll looks like from outside: the bars
    belong to the old contract, the price to the new one.
    """
    payload = fake_chart(bars, price)
    payload["chart"]["result"][0]["meta"]["regularMarketChangePercent"] = vendor_change
    return payload


# BZ=F on 2026-09-15: bars on the old contract (105.68 prior close), price
# on a new one trading ~$5 lower. Deriving change from the bars gives
# -2.27% on a session Brent was up 2.27% -- the sign itself is wrong.
try:
    macro.requests = with_response(rolled_chart(
        [(epoch(2026, 9, 14), 105.68), (epoch(2026, 9, 15), 103.30)],
        price=103.30, vendor_change=2.287))
    rolled = macro.fetch_level("BZ=F")
finally:
    macro.requests = real_requests

check(
    "18. a contract roll does not invert the sign of the day's change",
    rolled["change_pct"] is not None and rolled["change_pct"] > 0,
)

check(
    "19. the vendor's change is used verbatim across a roll",
    abs(rolled["change_pct"] - 2.287) < 1e-9,
)

# ES=F overnight: the daily bar closes on a different boundary than the
# contract settles on, so the bars read +0.32% on a session the future was
# down ~0.6% -- while cash SPX, right beside it on the strip, read -0.52%.
try:
    macro.requests = with_response(rolled_chart(
        [(epoch(2026, 9, 14), 7600.0), (epoch(2026, 9, 15), 7624.0)],
        price=7624.0, vendor_change=-0.569))
    overnight = macro.fetch_level("ES=F")
finally:
    macro.requests = real_requests

check(
    "20. an overnight future is measured against settlement, not its daily bar",
    overnight["change_pct"] is not None and abs(overnight["change_pct"] + 0.569) < 1e-9,
)

# The fallback has to survive, or every symbol Yahoo omits the field for
# silently loses its change instead of falling back to the bars.
try:
    macro.requests = with_response(fake_chart(
        [(epoch(2026, 9, 10), 100.0), (epoch(2026, 9, 11), 110.0)], price=110.0))
    no_vendor = macro.fetch_level("TEST")
finally:
    macro.requests = real_requests

check(
    "21. with no vendor change, the bar-derived figure is still used",
    no_vendor["change_pct"] is not None and abs(no_vendor["change_pct"] - 10.0) < 1e-9,
)

# A non-numeric or NaN vendor field must not poison the row.
for bad in ("n/a", float("nan"), None):
    try:
        macro.requests = with_response(rolled_chart(
            [(epoch(2026, 9, 10), 100.0), (epoch(2026, 9, 11), 110.0)],
            price=110.0, vendor_change=bad))
        junk = macro.fetch_level("TEST")
    finally:
        macro.requests = real_requests
    check(
        f"22. a junk vendor change ({bad!r}) falls back to the bars",
        junk is not None and junk["change_pct"] is not None
        and abs(junk["change_pct"] - 10.0) < 1e-9,
    )

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
