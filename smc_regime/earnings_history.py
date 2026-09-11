"""Append-only record of each ticker's NEXT earnings date, as it was known.

Yahoo publishes only the upcoming report date -- no source available to this
project carries historical ones (TradingView's earnings feed states outright
that past report dates are not available). That gap blocked a real measurement:
trades whose holding window spanned a company-specific event beat SPY on 48%
of occasions against 69% for the rest, costing 340 points of alpha, and the
tradeable question -- how much of that was a SCHEDULED report that could have
been avoided -- needs report dates the past does not give up.

It gives them up going forward. Tonight's "next report" is next month's "last
report", so capturing the field nightly accumulates the history that cannot be
bought. A date recorded here was, by construction, KNOWN BEFORE THE EVENT --
which is exactly the property a tradeable filter has to be tested against.
Reconstructing dates after the fact from price gaps could never prove that.

WRITTEN ON CHANGE, NOT NIGHTLY. Re-recording 415 unchanged rows every night
would add ~9 MB a year to say nothing; a row is appended only when a ticker's
next date (or its estimate flag) actually moves. The transitions are the
information: when a ticker's next date jumps from E1 to a later E2, E1 is when
it reported. See reported_dates().

`is_estimate` is carried because Yahoo flags dates it inferred from last
year's cadence, and those move. A filter tested against dates that were only
ever guesses would overstate what was knowable in advance.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import db as db_module

DEFAULT_PATH = Path("backtest_logs/earnings_history.jsonl")
_FIELDS = ["as_of", "ticker", "earnings_date", "is_estimate"]


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Every recorded transition, sorted by (ticker, as_of)."""
    if not path.exists():
        return pd.DataFrame(columns=_FIELDS)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame(columns=_FIELDS)
    return pd.DataFrame(rows).sort_values(["ticker", "as_of"]).reset_index(drop=True)


def latest(frame: pd.DataFrame) -> dict[str, tuple[str, bool | None]]:
    """ticker -> (earnings_date, is_estimate) as most recently recorded."""
    if frame.empty:
        return {}
    last = frame.sort_values("as_of").groupby("ticker").last()
    return {t: (r["earnings_date"], r["is_estimate"]) for t, r in last.iterrows()}


def changed_rows(current: dict[str, dict], known: dict[str, tuple], as_of: str) -> list[dict]:
    """Rows for tickers whose next date or estimate flag has moved.

    A ticker Yahoo returned nothing for is skipped rather than recorded as
    null: "we did not fetch it" and "it has no scheduled date" are different
    claims, and only the second belongs in a history.
    """
    records = []
    for ticker, data in sorted(current.items()):
        date = data.get("earnings_date")
        if not date:
            continue
        flag = data.get("earnings_is_estimate")
        flag = None if flag is None else bool(flag)
        if known.get(ticker) == (date, flag):
            continue
        records.append({"as_of": as_of, "ticker": ticker,
                        "earnings_date": date, "is_estimate": flag})
    return records


def append(records: list[dict], path: Path = DEFAULT_PATH) -> int:
    """Append rows, then normalise the whole file. Returns rows now held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps({k: record[k] for k in _FIELDS}) + "\n")
    return normalize(path)


def normalize(path: Path = DEFAULT_PATH) -> int:
    """Sort and de-duplicate on (ticker, as_of, earnings_date), in place.

    The nightly and any hand-run capture both append here, so the file gets
    the same union-merge treatment as the other logs -- which means it can
    arrive with both sides' rows interleaved and duplicated.
    """
    if not path.exists():
        return 0
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    markers = [ln for ln in lines if ln.startswith(("<<<<<<<", "=======", ">>>>>>>"))]
    if markers:
        raise ValueError(
            f"{path} still has {len(markers)} conflict marker line(s) -- the union merge "
            "driver did not apply. Check .gitattributes before re-running."
        )
    seen, out = set(), []
    for record in sorted((json.loads(ln) for ln in lines),
                         key=lambda r: (r["ticker"], r["as_of"], r["earnings_date"])):
        key = (record["ticker"], record["as_of"], record["earnings_date"])
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    path.write_text("".join(json.dumps({k: r[k] for k in _FIELDS}) + "\n" for r in out))
    return len(out)


def reported_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Historical report dates, derived from the transitions.

    A date stops being "next" because it happened. So when a ticker's recorded
    date moves from E1 to a LATER E2, E1 is a date it reported on, and the
    as_of of that change bounds when we learned it was done. A move to an
    EARLIER date is a correction, not a report, and is not counted.

    `confirmed` is False when the date was still flagged an estimate the last
    time it was seen -- it may have drifted before the company actually
    reported, so it is a weaker claim than a confirmed one.
    """
    out = []
    for ticker, g in frame.sort_values("as_of").groupby("ticker"):
        rows = g.to_dict("records")
        for prev, nxt in zip(rows, rows[1:]):
            if nxt["earnings_date"] > prev["earnings_date"]:
                out.append({"ticker": ticker, "reported_on": prev["earnings_date"],
                            "known_by": prev["as_of"], "observed_done": nxt["as_of"],
                            "confirmed": prev["is_estimate"] is False})
    return pd.DataFrame(out, columns=["ticker", "reported_on", "known_by", "observed_done", "confirmed"])


def capture(db_path: str = str(db_module.DEFAULT_DB_PATH),
            path: Path = DEFAULT_PATH, as_of: str | None = None) -> tuple[str, int, int]:
    """Record tonight's dates. Returns (as_of, rows_appended, rows_in_file)."""
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    conn = db_module.connect(db_path)
    valuation = db_module.all_valuation(conn)
    conn.close()
    current = {
        r["ticker"]: {"earnings_date": None if pd.isna(r["earnings_date"]) else r["earnings_date"],
                      "earnings_is_estimate": None if pd.isna(r["earnings_is_estimate"]) else r["earnings_is_estimate"]}
        for _, r in valuation.iterrows()
    }
    records = changed_rows(current, latest(load(path)), as_of)
    total = append(records, path) if records else normalize(path)
    return as_of, len(records), total


def main() -> None:
    """Default action is NORMALIZE, matching grade_history's CLI contract --
    the nightly's rebase-retry loop calls both the same way after a union
    merge, and capturing there instead would re-read a database that may
    already have moved on."""
    parser = argparse.ArgumentParser(description="Normalize the earnings-date history in place.")
    parser.add_argument("--history-file", default=str(DEFAULT_PATH))
    parser.add_argument("--capture", action="store_true", help="record tonight's dates from the database")
    parser.add_argument("--report", action="store_true", help="print the report dates derived from transitions")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()

    path = Path(args.history_file)
    if args.capture:
        as_of, appended, total = capture(args.db_file, path, args.as_of)
        print(f"{as_of}: {appended} earnings date(s) changed; history holds {total} transitions")
        return
    if args.report:
        frame = load(path)
        done = reported_dates(frame)
        print(f"{len(done)} report dates derived from {len(frame)} transitions")
        if not done.empty:
            print(done.tail(20).to_string(index=False))
        return
    if not path.exists():
        print(f"{path}: not found, nothing to normalize", file=sys.stderr)
        return
    print(f"{path}: {normalize(path)} transitions")


if __name__ == "__main__":
    main()
