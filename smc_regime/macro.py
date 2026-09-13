"""Macro context for the setup dashboard: market levels and scheduled events.

Two halves with very different reliability, kept separate on purpose.

LEVELS come from Yahoo's chart endpoint -- the same un-gated API data.py
already uses for OHLCV, not the cookie+crumb quoteSummary flow valuation.py
depends on. That matters: this runs in the same nightly, and adding a second
call against the brittle endpoint is how a 400-ticker batch starts getting
rate-limited.

EVENTS come from a committed file, because the nightly runs in GitHub
Actions where no MCP server is available, and the primary sources
(federalreserve.gov, bls.gov) are not reachable from the dev environment
either. The file is seeded from a calendar feed and checked in, the same
pattern as data/exchange_cache.json. It is NOT refreshed by the nightly.

THE CALENDAR EXPIRES. The feed it is seeded from only publishes about a
month ahead, so this needs topping up monthly -- not annually, as the
publication schedules of the FOMC and BLS might suggest. A stale calendar
is worse than none: "no Fed meeting scheduled" reads as reassurance when
it actually means nobody refreshed the file. So `covers_through` is
carried all the way to the page, every consumer is expected to show it,
and calendar_expired() exists to make the check hard to skip.

NOTHING HERE IS SCORED. It is context for reading a setup, exactly like
news sentiment and the earnings countdown -- a Fed day does not make a
B setup an A, it makes it a B setup you are taking into a Fed day.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_CALENDAR_PATH = Path(__file__).parent / "data" / "macro_calendar.json"
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 20

# Symbol -> (display label, unit). Ordered: the page renders them in this
# order, slowest-moving first, because the page is baked at publish time
# and the slow ones are the ones still worth reading hours later.
LEVELS = [
    ("^TNX", "US 10Y", "%"),
    ("CL=F", "WTI crude", "$"),
    ("BZ=F", "Brent", "$"),
    ("GC=F", "Gold", "$"),
    ("DX-Y.NYB", "Dollar index", ""),
    ("^VIX", "VIX", ""),
    ("ES=F", "S&P futures", ""),
    ("NQ=F", "Nasdaq futures", ""),
]


def fetch_level(symbol: str) -> dict | None:
    """Last close and its move against the prior close, or None on failure.

    Deliberately returns None rather than raising: a macro strip is context,
    and one unreachable symbol should leave a gap on the page, not take down
    a nightly that has already done four hours of real work.
    """
    try:
        response = requests.get(
            _CHART_URL.format(symbol=symbol),
            params={"range": "5d", "interval": "1d"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        chart = response.json()["chart"]
        if chart.get("error"):
            return None
        result = chart["result"][0]
        closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
        if not closes:
            return None
        last = result["meta"].get("regularMarketPrice") or closes[-1]
        prior = closes[-2] if len(closes) > 1 else None
        return {
            "symbol": symbol,
            "price": float(last),
            # None, not 0.0, when there is no prior close to compare against:
            # a flat reading and an unknown one are different claims.
            "change_pct": None if prior in (None, 0) else (float(last) / prior - 1) * 100,
        }
    except Exception:
        return None


def fetch_levels(symbols=None) -> list[dict]:
    """Every configured level, in display order, skipping any that failed."""
    wanted = LEVELS if symbols is None else [row for row in LEVELS if row[0] in symbols]
    out = []
    for symbol, label, unit in wanted:
        row = fetch_level(symbol)
        if row is not None:
            out.append({**row, "label": label, "unit": unit})
    return out


def load_calendar(path: Path = _CALENDAR_PATH) -> dict:
    """The committed event schedule. Empty but well-formed when absent, so a
    checkout without the file renders an honest "no calendar" rather than
    raising four hours into a run."""
    if not Path(path).exists():
        return {"events": [], "covers_through": None, "source": None}
    with open(path) as f:
        return json.load(f)


def calendar_expired(calendar: dict, now: datetime | None = None) -> bool:
    """True once the schedule no longer covers today.

    The point of asking is that an expired calendar shows NOTHING, which
    looks identical to a genuinely quiet week. Callers are expected to say
    which one it is.
    """
    through = calendar.get("covers_through")
    if not through:
        return True
    now = now or datetime.now(timezone.utc)
    return now.date() > datetime.fromisoformat(through).date()


def upcoming(calendar: dict, now: datetime | None = None, days: int = 14) -> list[dict]:
    """Events from now through `days` ahead, soonest first.

    Same-day same-category events collapse to one row: a CPI print posts
    four separate lines (headline and core, MoM and YoY) that are one
    event to anyone reading a dashboard.
    """
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(days=days)
    seen, out = set(), []
    for event in sorted(calendar.get("events", []), key=lambda e: e["at"]):
        at = datetime.fromisoformat(event["at"].replace("Z", "+00:00"))
        if at < now - timedelta(hours=12) or at > horizon:
            continue
        key = (at.date(), event["category"])
        if event["category"] != "other" and key in seen:
            continue
        seen.add(key)
        out.append({**event})
    return out


def snapshot(days: int = 14) -> dict:
    """Everything the dashboard needs, in one call."""
    calendar = load_calendar()
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "levels": fetch_levels(),
        "events": upcoming(calendar, days=days),
        "covers_through": calendar.get("covers_through"),
        "expired": calendar_expired(calendar),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    snap = snapshot(days=args.days)
    for level in snap["levels"]:
        change = "   n/a" if level["change_pct"] is None else f"{level['change_pct']:+6.2f}%"
        print(f"  {level['label']:<16} {level['price']:>10,.2f} {change}")
    print(f"\n  calendar through {snap['covers_through']}" + ("  [EXPIRED]" if snap["expired"] else ""))
    for event in snap["events"]:
        print(f"    {event['at'][:10]}  {event['category']:<10} {event['title']}")


if __name__ == "__main__":
    main()
