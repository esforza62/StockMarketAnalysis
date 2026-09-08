"""Append-only history of per-ticker setup grades, so a grade can be
checked against what the ticker actually did afterwards.

Nothing else in this project retains that. The nightly workflow builds the
SQLite store from scratch on every run (it never downloads the previous
one), `valuation` is a plain upsert keyed on ticker, and the grades
themselves were never persisted at all -- they were derived at export time
and existed only inside the published artifact. So "what did this score on
1 September, and how has it done since" had no answer.

Two deliberate choices:

* Keyed on a DATE, never on `runs.id`. That column is `INTEGER PRIMARY KEY`
  with no AUTOINCREMENT and `clear_trades()` empties the table, so run ids
  restart at 1 on every rebuild -- anything keyed on one silently comes to
  mean a different run.

* A committed JSONL file rather than a table, following
  regime_strategy_log.jsonl. A table would be wiped nightly along with the
  rest of the database unless the workflow started fetching the previous
  copy first; a file in the repo survives on its own.

`close` is stored with every row because it is the anchor the whole
exercise depends on: forward return over any horizon is a later row's close
over this row's, both read back out of this same file. That keeps one
mechanism for the backfill and for nightly capture, with no separate price
source to drift.

At 415 tickers this is ~166KB of appended text per captured day. Git deltas
that down to ~10KB per commit, so repo growth is a few MB a year, but the
working file itself reaches ~40MB over a year of daily capture -- worth
revisiting (rotate by year, or drop the component columns for old dates)
once there is enough history to have answered whether the grades work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("backtest_logs/setup_grade_history.jsonl")

# Grade plus the inputs needed to interpret it later. The component points
# are kept (they are what a post-mortem asks about -- "did the A names win
# on trend structure or on volume?") but the prose `*_detail` strings are
# not: they are the largest field by far and reconstructible from the
# points, and this file is committed on every run.
_COMPONENT_FIELDS = [
    "trend_structure_points",
    "rsi_points",
    "macd_points",
    "volume_points",
    "streak_points",
    "alignment_points",
    "sector_industry_points",
    "valuation_points",
]

_BASE_FIELDS = ["ticker", "sector", "grade", "total_points", "close", "regime", "direction"]


def to_records(
    scores: pd.DataFrame,
    as_of: str,
    interval: str,
    closes: dict[str, float] | None = None,
    valuation_available: bool = True,
) -> list[dict]:
    """Turn a scored frame into history rows for one as-of date.

    `closes` supplies the price the grade was struck at when the frame does
    not already carry one. `valuation_available` is recorded per row rather
    than inferred: a backfilled grade scores valuation at its neutral half
    credit because historical P/E cannot be reconstructed, and a report that
    cannot see that would compare it against live grades as if they were
    measured the same way.
    """
    records = []
    for _, r in scores.iterrows():
        close = r["close"] if "close" in r and pd.notna(r.get("close")) else (closes or {}).get(r["ticker"])
        record = {"as_of": as_of, "interval": interval}
        for field in _BASE_FIELDS:
            value = close if field == "close" else r.get(field)
            if isinstance(value, float):
                value = round(value, 4)
            record[field] = None if value is not None and pd.isna(value) else value
        for field in _COMPONENT_FIELDS:
            record[field] = r.get(field)
        record["valuation_available"] = bool(valuation_available)
        records.append(record)
    return records


def append(records: list[dict], path: Path = DEFAULT_PATH) -> int:
    """Append rows, then normalise the whole file. Returns rows now held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return normalize(path)


def normalize(path: Path = DEFAULT_PATH) -> int:
    """Deduplicate on (as_of, interval, ticker) and sort. Last write wins,
    so re-running a date corrects it rather than duplicating it.

    Also the post-merge repair step: like the snapshot log, this file is
    `merge=union` in .gitattributes so two overlapping runs both keep their
    rows instead of conflicting on every appended line, and union leaves
    the result unsorted and possibly duplicated.
    """
    original = path.read_text()
    lines = [ln for ln in original.splitlines() if ln.strip()]

    # Only reachable if the union merge driver did not apply -- a checkout
    # whose .gitattributes predates it. Say so, rather than letting a raw
    # JSON decode error suggest the history itself is corrupt.
    markers = [ln for ln in lines if ln.startswith(("<<<<<<<", "=======", ">>>>>>>"))]
    if markers:
        raise ValueError(
            f"{path} still has {len(markers)} conflict marker line(s) -- the union merge "
            "driver did not apply. Check that .gitattributes is present in the tree being "
            "merged into, then resolve the conflict before normalizing."
        )

    deduped: dict[tuple, dict] = {}
    for line in lines:
        r = json.loads(line)
        deduped[(r["as_of"], r["interval"], r["ticker"])] = r

    ordered = sorted(deduped.values(), key=lambda r: (r["as_of"], r["interval"], r["ticker"]))
    rebuilt = "".join(json.dumps(r) + "\n" for r in ordered)
    # Leave an already-normalized file untouched, so the workflow can use
    # `git diff --quiet` to tell whether there is anything to amend.
    if rebuilt != original:
        path.write_text(rebuilt)
    return len(ordered)


def load(path: Path = DEFAULT_PATH, interval: str | None = None) -> pd.DataFrame:
    """Read the history back. Empty frame (with the right columns) when the
    file does not exist yet, so callers do not special-case a first run."""
    columns = ["as_of", "interval", *_BASE_FIELDS, *_COMPONENT_FIELDS, "valuation_available"]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)

    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    if not df.empty and interval is not None:
        df = df[df["interval"] == interval].reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the setup-grade history in place.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"{path}: not found, nothing to normalize", file=sys.stderr)
        return
    print(f"{path}: {normalize(path)} rows")


if __name__ == "__main__":
    main()
