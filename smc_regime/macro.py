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
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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
]

# Equity benchmarks, resolved by SESSION rather than fixed.
#
# During the cash session the index itself is the market -- SPX at 2pm is
# the number everything on this page is measured against. Overnight and
# pre-market the index is frozen at yesterday's close and the future is the
# only thing moving, so showing the cash index at 3am would print a stale
# number with no indication it was stale.
#
# The switch is DATA-DRIVEN, not clock-driven: the cash index is used when
# its own latest daily bar is dated today in exchange time, which means it
# has actually traded. That gets market holidays and half-days right
# without carrying a holiday calendar, and degrades to the future on any
# day the cash market did not open.
INDEX_PAIRS = [
    (("^GSPC", "S&P 500"), ("ES=F", "S&P futures")),
    (("^NDX", "Nasdaq 100"), ("NQ=F", "Nasdaq futures")),
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
        meta = result["meta"]
        quote = result["indicators"]["quote"][0]
        bars = [
            (stamp, close)
            for stamp, close in zip(result.get("timestamp") or [], quote.get("close") or [])
            if close is not None
        ]
        if not bars:
            return None
        last = meta.get("regularMarketPrice") or bars[-1][1]

        # The prior close is the last bar dated BEFORE today in exchange
        # time -- not simply bars[-2]. Mid-session Yahoo may or may not have
        # opened today's daily bar yet, and taking the second-to-last either
        # way silently compares the live price against the day before
        # yesterday on the runs that happen to land before it appears.
        # chartPreviousClose is no help: it is the close preceding the whole
        # requested range, not the previous session.
        zone = _zone(meta.get("exchangeTimezoneName"))
        today = datetime.now(zone).date()
        # Anchor on the LATEST BAR's date, not on today. When the market is
        # shut the latest bar already carries the current price, so asking
        # for "the last bar before today" hands back that same bar and every
        # change reads a flat 0.00% -- which is what the first version of
        # this did, over a weekend, for all eight instruments at once.
        latest_date = datetime.fromtimestamp(bars[-1][0], zone).date()
        earlier = [c for stamp, c in bars if datetime.fromtimestamp(stamp, zone).date() < latest_date]
        prior = earlier[-1] if earlier else None

        # PREFER YAHOO'S OWN CHANGE over one derived from the daily bars.
        # The bar series is spliced, not a single instrument, and the two
        # ways it lies are both invisible in the number itself:
        #
        #   * A FUTURES ROLL swaps the contract underneath the series. On
        #     2026-09-15 BZ=F rolled to a month trading ~$5 lower in a
        #     backwardated curve, so today's price over yesterday's close
        #     read -2.27% on a session Brent was UP 2.27%. The sign was
        #     wrong, next to WTI +3.2%, and nothing about the bars said so.
        #   * A 23-HOUR FUTURE settles on a different boundary than its
        #     daily bar closes on, so ES=F/NQ=F read +0.32%/+0.35% against
        #     the bars while the vendor -- and cash SPX at -0.52% -- had
        #     them down ~0.6%. That is the overnight strip's headline pair.
        #
        # Yahoo computes this against the right reference in both cases.
        # Across the ten configured symbols the two methods agree to three
        # decimals on the seven that are neither rolling nor overnight
        # futures, so this is a no-op except where the bars are wrong.
        derived = None if prior in (None, 0) else (float(last) / prior - 1) * 100
        vendor = meta.get("regularMarketChangePercent")
        try:
            vendor = None if vendor is None else float(vendor)
        except (TypeError, ValueError):
            vendor = None
        if vendor is not None and vendor != vendor:  # NaN
            vendor = None

        return {
            "symbol": symbol,
            "price": float(last),
            # None, not 0.0, when there is neither a vendor figure nor a
            # prior close to compare against: a flat reading and an unknown
            # one are different claims.
            "change_pct": vendor if vendor is not None else derived,
            # Whether this instrument has traded today in its own timezone,
            # which is what decides cash-versus-future below.
            "traded_today": bool(bars and datetime.fromtimestamp(bars[-1][0], zone).date() >= today),
            # Which trading session the price and change above belong to.
            # Compared against the rest of the strip downstream, so a
            # future trading tomorrow's session can be labelled as such.
            "session_date": (
                lambda d: d.isoformat() if d else None
            )(_session_date(meta, zone, bars)),
        }
    except Exception:
        return None


# CME Globex reopens at 18:00 exchange-local, and everything on this strip
# that is a future (crude, Brent, gold, the equity futures) is on that
# schedule. After the reopen a future's "today" is the NEXT trading date,
# while the cash indices beside it are still showing the session that just
# closed.
_FUTURES_REOPEN_HOUR = 18


def _session_date(meta: dict, zone, bars: list) -> "date | None":
    """The trading date the current quote belongs to.

    For a future after the evening reopen this is tomorrow, not today: at
    8pm ET crude is trading the next session, so its change is measured
    against a settlement the cash market has not reached yet. The strip
    needs to say which session each number is from, or WTI reads -0.4% on
    an afternoon it settled up 4%.
    """
    stamp = meta.get("regularMarketTime")
    if meta.get("instrumentType") == "FUTURE" and stamp:
        local = datetime.fromtimestamp(stamp, zone)
        if local.hour >= _FUTURES_REOPEN_HOUR:
            nxt = local.date() + timedelta(days=1)
            while nxt.weekday() >= 5:  # Sunday reopen belongs to Monday
                nxt += timedelta(days=1)
            return nxt
        return local.date()
    if bars:
        return datetime.fromtimestamp(bars[-1][0], zone).date()
    return None


def _zone(name: str | None):
    """Exchange timezone, falling back to UTC when Yahoo omits or misnames it."""
    try:
        return ZoneInfo(name) if name else timezone.utc
    except Exception:
        return timezone.utc


def fetch_levels(symbols=None) -> list[dict]:
    """Every configured level, in display order, skipping any that failed.

    Equity benchmarks resolve per session: the cash index when it has
    traded today, otherwise the front future. Only one of each pair is
    returned, so the strip stays the same width either way.
    """
    wanted = LEVELS if symbols is None else [row for row in LEVELS if row[0] in symbols]
    out = []
    for symbol, label, unit in wanted:
        row = fetch_level(symbol)
        if row is not None:
            out.append({**row, "label": label, "unit": unit})

    if symbols is None:
        for (cash_symbol, cash_label), (fut_symbol, fut_label) in INDEX_PAIRS:
            cash = fetch_level(cash_symbol)
            if cash is not None and cash.get("traded_today"):
                out.append({**cash, "label": cash_label, "unit": ""})
                continue
            future = fetch_level(fut_symbol)
            if future is not None:
                # Named as a future so the label itself carries the caveat --
                # nobody should have to read the timestamp to work out that
                # a 3am number is not the index.
                out.append({**future, "label": fut_label, "unit": ""})
            elif cash is not None:
                out.append({**cash, "label": cash_label + " (prev close)", "unit": ""})
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
