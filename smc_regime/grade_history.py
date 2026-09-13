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

# The headline sentiment reading, kept as the NUMBER and the sample size,
# never the band label. The label is a function of the number under bands
# calibrated to one night's corpus (see news.py); storing it would freeze a
# calibration into the history and make a later recalibration unmeasurable
# against its own past. The count is what separates a real tail member from
# a ticker with two articles, and is the first thing a drift check needs.
#
# Sentiment is never scored, so this is here for one purpose: the nightly
# rebuilds the database from scratch and `news_sentiment` is overwritten
# every run, which means nothing retained the spread run to run. Without
# these two fields a question like "have the bands held since they were
# calibrated" can only ever be answered on the single night you happen to
# ask it.
_NEWS_FIELDS = ["news_compound", "news_articles"]


def _news_fields(row) -> dict:
    """Pull the sentiment reading out of a scored row's `news` dict.

    All None when the frame carries no news at all -- which is the backfill,
    where historical headlines cannot be reconstructed, the same reason
    valuation is flagged there rather than invented.
    """
    news = row.get("news") if "news" in row else None
    if not isinstance(news, dict):
        return dict.fromkeys(_NEWS_FIELDS)
    compound = news.get("avg_compound")
    return {
        "news_compound": None if compound is None or pd.isna(compound) else round(float(compound), 4),
        "news_articles": None if news.get("article_count") is None else int(news["article_count"]),
    }


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
        record.update(_news_fields(r))
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
    columns = ["as_of", "interval", *_BASE_FIELDS, *_COMPONENT_FIELDS, "valuation_available", *_NEWS_FIELDS]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)

    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    # Rows written before a field existed simply lack the key, so the frame
    # would come back a column short and a caller reading it would raise on
    # a name that is present in every other code path. Reindex instead: the
    # older rows read as NaN, which is the truth about them.
    if not df.empty:
        df = df.reindex(columns=[*columns, *(c for c in df.columns if c not in columns)])
    if not df.empty and interval is not None:
        df = df[df["interval"] == interval].reset_index(drop=True)
    return df


def grade_streaks(path: Path = DEFAULT_PATH, interval: str = "1d") -> dict[str, dict]:
    """How long each ticker has held its current grade, per ticker.

    Returns {ticker: {grade, since, observations, strict_observations,
    total_at_grade, captures, flips, censored}}. `since` is the earliest
    captured date the current grade runs back to, tolerating isolated
    single-capture disagreements; `flips` lists any it crossed.

    TWO THINGS MAKE THE NAIVE COUNT WRONG, and both of them read as
    confidence rather than as error.

    Re-filed sessions. A capture whose close is identical to the previous
    one for that ticker is the SAME trading session filed under a new date
    -- it happens when a snapshot runs on a weekend or a market holiday,
    and it happened for all 415 tickers when a run was dispatched on a
    Saturday night carrying Friday's closes. Counting it adds a day nobody
    lived through, so consecutive identical closes collapse to one
    observation. Grade is deliberately not part of that test: two genuinely
    different sessions can easily produce the same grade, and collapsing
    those would be the opposite error.

    Censoring. The history starts when capture started, so a ticker that
    has held its grade for every capture on file has a streak of AT LEAST
    that long and possibly far longer. `censored` says so, and callers are
    expected to render it as a floor rather than a measurement -- at the
    time of writing 40 of 415 tickers sit at that boundary, so it is the
    common case, not an edge one.

    Gaps are NOT treated as breaks. If a grade reads A either side of a
    missed run, the streak continues through it: the alternative resets a
    real streak every time the scheduler skips a night, which would be
    wrong far more often than the continuity assumption is.
    """
    frame = load(path, interval=interval)
    if frame.empty:
        return {}

    out: dict[str, dict] = {}
    for ticker, rows in frame.groupby("ticker"):
        ordered = rows.sort_values("as_of")
        captures: list[tuple[str, str]] = []   # (as_of, grade), re-files collapsed
        last_close = None
        for row in ordered.itertuples(index=False):
            close = getattr(row, "close", None)
            if captures and close is not None and last_close is not None and close == last_close:
                # Same session re-filed: replace rather than append, so the
                # streak is dated from when the session actually happened.
                captures[-1] = (captures[-1][0], row.grade)
                continue
            captures.append((row.as_of, row.grade))
            last_close = close
        if not captures:
            continue

        current = captures[-1][1]
        # Walk back tolerating ISOLATED disagreements. A strict streak
        # answers the wrong question here: scores bunch within a point of
        # the grade cuts (hence the borderline tags on the dashboard), so a
        # name sitting a fraction from a boundary flips on noise and resets
        # to zero. On the first history deep enough to check, every one of
        # 415 tickers had its current grade interrupted at least once, which
        # made the strict number read as "days since the last wobble"
        # rather than "how settled this grade is".
        #
        # Isolated means a single capture with the same grade on BOTH sides.
        # Two in a row is a real change, and a disagreement at the far end
        # of the history has nothing before it to confirm continuity, so
        # neither is absorbed. Every one crossed is counted and returned in
        # `flips`, because a streak that quietly swallowed three reversals
        # would be the same false confidence the strict count avoided.
        flips, since, observations, index = [], captures[-1][0], 0, len(captures) - 1
        while index >= 0:
            as_of, grade = captures[index]
            if grade == current:
                observations += 1
                since = as_of
                index -= 1
                continue
            isolated = index > 0 and captures[index - 1][1] == current
            if not isolated:
                break
            flips.append({"at": as_of, "grade": grade})
            index -= 1
        out[ticker] = {
            "grade": current,
            "since": since,
            "observations": observations,
            "flips": list(reversed(flips)),
            # `strict` is the unbroken run, kept so the page can say what
            # the tolerance actually bought and a reader can tell a clean
            # streak from a patched one.
            "strict_observations": _strict_run(captures, current),
            "total_at_grade": sum(1 for _, g in captures if g == current),
            "captures": len(captures),
            "censored": index < 0,
        }
    return out


def _strict_run(captures: list[tuple[str, str]], current: str) -> int:
    """Consecutive captures at `current`, counting back with no tolerance."""
    n = 0
    for _as_of, grade in reversed(captures):
        if grade != current:
            break
        n += 1
    return n


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
