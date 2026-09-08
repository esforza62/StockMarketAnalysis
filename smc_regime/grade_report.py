"""Did the grades work? Forward returns of each grade, out of the history.

This is the question the grade history exists to answer: A-graded names are
supposed to outperform D-graded ones over the following weeks, and until
now there was no way to check whether they did. Every number here comes out
of `setup_grade_history.jsonl` -- the grade and the close it was struck at
on one date, against the close on a later date -- so there is no second
price source that could disagree with what the grade was computed from.

Reading the output
------------------
The headline is not the absolute return of any grade. It is the *ordering*:
over a horizon, does mean return fall monotonically from A to D? A regime
tool earns its keep by ranking, and an A that trails its D bucket is a
finding regardless of whether both happened to be positive in a rising
market. `spread` (A minus D) is that ordering in one number.

Two caveats the report prints rather than hides:

* Buckets are unequal and often small. A day with four A grades gives a
  mean over four names; treat one date's spread as an anecdote. `n` is
  shown on every cell for that reason.
* Backfilled rows scored valuation at neutral half credit because
  historical P/E does not exist (see grade_backfill). Where a horizon mixes
  backfilled and live rows the report says so, because the grade cut moved
  under those rows and the same ticker could land in different buckets on
  either side of the join.

A horizon that has not elapsed yet is reported as pending, never as a
missing or zero return -- "no 1-month number yet" and "a 1-month return of
nothing" are very different claims.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import grade_history
from . import jsonfmt

_GRADES = ["A", "B", "C", "D"]

# Calendar days, not rows. A row count would mean "5 rows later", which is
# five trading days only while the history has no gaps -- one skipped
# nightly run and a "1 week" return would quietly become nine days.
DEFAULT_HORIZONS = {"1w": 7, "2w": 14, "1m": 30, "3m": 91}

# How far past the target date a row may sit and still count. A holiday
# weekend can push the next available bar three days out; anything beyond
# this is a gap in the history, and stretching a 7-day return over three
# weeks would misreport it rather than fill it in.
_HORIZON_SLACK_DAYS = 5


def forward_returns(
    history: pd.DataFrame,
    horizons: dict[str, int] = DEFAULT_HORIZONS,
    slack_days: int = _HORIZON_SLACK_DAYS,
) -> pd.DataFrame:
    """Attach a forward return per horizon to every history row.

    The later price is the ticker's own next recorded close at or after
    `as_of + days`, within `slack_days`. NaN means the horizon has not
    elapsed yet or the history has a hole there -- the two are separated by
    `maturity`, since only the first is a reason to keep waiting.
    """
    df = history.copy()
    if df.empty:
        return df.assign(**{f"fwd_{h}": pd.Series(dtype=float) for h in horizons})

    df["as_of_dt"] = pd.to_datetime(df["as_of"])
    latest = df["as_of_dt"].max()

    for name, days in horizons.items():
        df[f"fwd_{name}"] = pd.NA
        df[f"mature_{name}"] = False

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("as_of_dt")
        dates = group["as_of_dt"].to_numpy()
        closes = group["close"].to_numpy()
        for name, days in horizons.items():
            target = group["as_of_dt"] + pd.Timedelta(days=days)
            # Whether the horizon has elapsed at all is a property of the
            # calendar, not of whether a row happens to exist for it.
            df.loc[group.index, f"mature_{name}"] = (target <= latest).to_numpy()
            pos = dates.searchsorted(target.to_numpy(), side="left")
            for i, (idx, p) in enumerate(zip(group.index, pos)):
                if p >= len(dates):
                    continue
                gap = (dates[p] - dates[i]) / pd.Timedelta(days=1)
                if gap > days + slack_days:
                    continue
                start, end = closes[i], closes[p]
                if pd.isna(start) or pd.isna(end) or not start:
                    continue
                df.at[idx, f"fwd_{name}"] = (end - start) / start * 100

    for name in horizons:
        df[f"fwd_{name}"] = pd.to_numeric(df[f"fwd_{name}"], errors="coerce")
    return df


def summarize(
    scored: pd.DataFrame,
    horizons: dict[str, int] = DEFAULT_HORIZONS,
) -> dict:
    """Per-grade forward performance, plus whether the ordering held."""
    out: dict = {"horizons": {}, "row_count": int(len(scored))}
    if scored.empty:
        return out

    for name in horizons:
        col, mature_col = f"fwd_{name}", f"mature_{name}"
        matured = scored[scored[mature_col]]
        usable = matured[matured[col].notna()]

        buckets = {}
        for grade in _GRADES:
            rows = usable[usable["grade"] == grade][col]
            buckets[grade] = {
                "n": int(len(rows)),
                "mean_pct": round(float(rows.mean()), 2) if len(rows) else None,
                "median_pct": round(float(rows.median()), 2) if len(rows) else None,
                "win_rate_pct": round(float((rows > 0).mean() * 100), 1) if len(rows) else None,
            }

        means = [buckets[g]["mean_pct"] for g in _GRADES]
        ranked = [m for m in means if m is not None]
        spread = (
            round(buckets["A"]["mean_pct"] - buckets["D"]["mean_pct"], 2)
            if buckets["A"]["mean_pct"] is not None and buckets["D"]["mean_pct"] is not None
            else None
        )
        out["horizons"][name] = {
            "days": horizons[name],
            "grades": buckets,
            # Whether the tool ranks at all. Monotonic means every step down
            # the grades gave up return, which is the claim being tested.
            # None -- not False -- when fewer than two buckets have data:
            # "nothing has matured yet" is not the same finding as "the
            # ordering came out wrong", and reporting it as the latter would
            # be a false alarm on every fresh history.
            "monotonic": all(a >= b for a, b in zip(ranked, ranked[1:])) if len(ranked) > 1 else None,
            "spread_a_minus_d_pct": spread,
            "pending_rows": int((~scored[mature_col]).sum()),
            "gap_rows": int((scored[mature_col] & scored[col].isna()).sum()),
        }

    valuation = scored["valuation_available"]
    out["valuation_mix"] = {
        "live_rows": int(valuation.sum()),
        "backfilled_rows": int((~valuation).sum()),
        # Grades either side of this join were struck with valuation
        # measured differently, so the buckets are not strictly like-for-like.
        "mixed": bool(valuation.any() and (~valuation).any()),
    }
    out["date_range"] = [scored["as_of"].min(), scored["as_of"].max()]
    out["dates"] = int(scored["as_of"].nunique())
    return out


def by_date(scored: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """One row per as-of date: how each grade did over `horizon` from that
    date. Shows whether an edge is persistent or one good week."""
    col = f"fwd_{horizon}"
    usable = scored[scored[f"mature_{horizon}"] & scored[col].notna()]
    if usable.empty:
        return pd.DataFrame()
    table = usable.pivot_table(index="as_of", columns="grade", values=col, aggfunc="mean").round(2)
    counts = usable.pivot_table(index="as_of", columns="grade", values=col, aggfunc="size")
    for grade in _GRADES:
        if grade not in table.columns:
            table[grade] = pd.NA
            counts[grade] = 0
    table = table[_GRADES]
    table["spread"] = (table["A"] - table["D"]).round(2)
    table["n"] = counts[_GRADES].sum(axis=1).astype(int)
    return table.reset_index()


def _print_report(summary: dict, per_date: pd.DataFrame | None, horizon: str) -> None:
    if not summary.get("horizons"):
        print("No graded history yet -- run smc_regime.grade_backfill or a nightly capture first.")
        return

    lo, hi = summary["date_range"]
    print(f"{summary['row_count']} graded rows over {summary['dates']} date(s), {lo} to {hi}")
    mix = summary["valuation_mix"]
    if mix["mixed"]:
        print(
            f"  note: {mix['backfilled_rows']} backfilled row(s) scored valuation at neutral half "
            f"credit and {mix['live_rows']} live row(s) did not -- buckets are not strictly "
            "like-for-like across that join"
        )
    elif mix["backfilled_rows"]:
        print("  note: all rows backfilled; valuation held neutral, so grades run ~5 points flatter than live")

    for name, h in summary["horizons"].items():
        print(f"\n  {name} ({h['days']} calendar days)")
        print(f"    {'grade':>6} {'n':>5} {'mean %':>8} {'median %':>9} {'win %':>7}")
        for grade in _GRADES:
            b = h["grades"][grade]
            if not b["n"]:
                print(f"    {grade:>6} {0:>5}        --        --      --")
                continue
            print(f"    {grade:>6} {b['n']:>5} {b['mean_pct']:>8.2f} {b['median_pct']:>9.2f} {b['win_rate_pct']:>7.1f}")
        if h["monotonic"] is None:
            print("    not enough matured rows to judge the ordering yet")
        else:
            spread = "--" if h["spread_a_minus_d_pct"] is None else f"{h['spread_a_minus_d_pct']:+.2f}%"
            verdict = "A>=B>=C>=D holds" if h["monotonic"] else "ordering BROKEN"
            print(f"    A-D spread {spread}, {verdict}")
        if h["pending_rows"]:
            print(f"    {h['pending_rows']} row(s) pending -- horizon not elapsed yet")
        if h["gap_rows"]:
            print(f"    {h['gap_rows']} row(s) unmeasurable -- no close within {_HORIZON_SLACK_DAYS} days of target")

    if per_date is not None and not per_date.empty:
        print(f"\n  {horizon} return by date")
        print("    " + per_date.to_string(index=False).replace("\n", "\n    "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-file", default=str(grade_history.DEFAULT_PATH))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--as-of", default=None, help="only grades struck on this date")
    parser.add_argument("--grade", default=None, help="only this grade, e.g. A")
    parser.add_argument("--horizon", default="1m", help=f"which horizon the by-date table uses ({'/'.join(DEFAULT_HORIZONS)})")
    parser.add_argument("--by-date", action="store_true", help="also print per-date performance")
    parser.add_argument("--tickers", action="store_true", help="list the individual names instead of summarising")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args()

    history = grade_history.load(Path(args.history_file), interval=args.interval)
    if history.empty:
        print(f"{args.history_file}: no history for interval {args.interval}", file=sys.stderr)
        return

    scored = forward_returns(history)
    if args.as_of:
        scored = scored[scored["as_of"] == args.as_of]
    if args.grade:
        scored = scored[scored["grade"] == args.grade.upper()]
    if scored.empty:
        print("no rows match those filters", file=sys.stderr)
        return

    if args.tickers:
        cols = ["as_of", "ticker", "sector", "grade", "total_points", "close"] + [
            f"fwd_{h}" for h in DEFAULT_HORIZONS
        ]
        listing = scored[cols].sort_values(["as_of", "total_points"], ascending=[True, False])
        print(listing.to_string(index=False, na_rep="--"))
        return

    summary = summarize(scored)
    if args.json:
        print(jsonfmt.dumps(summary, compact_depth=3))
        return
    _print_report(summary, by_date(scored, args.horizon) if args.by_date else None, args.horizon)


if __name__ == "__main__":
    main()
