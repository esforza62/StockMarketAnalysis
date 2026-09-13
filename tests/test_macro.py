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

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
