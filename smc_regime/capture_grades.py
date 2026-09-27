"""Append tonight's live grades to the grade history, after the snapshot.

Runs as its own step rather than inside daily_snapshot because it is a
read of what the snapshot produced, not part of producing it: if grading
breaks, the snapshot and the database it publishes are still good, and the
failure is visible on its own step instead of taking the whole run down.

These rows are the live counterpart to grade_backfill's: same universe,
same scoring function, but with real trailing/forward P/E, so they are
written with `valuation_available: true`. grade_report keeps the two apart
because a backfilled grade held valuation at neutral half credit and the
grade cut therefore sat differently underneath it.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import db as db_module
from . import grade_history
from .setup_score import compute_universe_setup_scores


def _run_date(conn, interval: str) -> str | None:
    """The date to file tonight's grades under: the date of the LAST BAR the
    run analysed, which is the session the grades actually describe.

    THIS USED TO BE THE RUN'S WALL-CLOCK DATE, on the reasoning that the
    scheduled run fires at 21:00 UTC and its UTC date is therefore the
    session date. That stopped being true: GitHub's scheduler drifted, and
    the nightly now starts between 23:11 and 23:59 UTC, so the 1d step
    finishes on either side of midnight unpredictably. Four of six runs
    filed wrongly before this was caught --

        run 38  1d finished 00:11  Mon session filed as Tue
        run 39  1d finished 23:47  filed as Tue too, COLLIDING with 38,
                                   and the duplicate was deduped away
        run 41  1d finished 00:10  Thu session filed as Fri
        run 42  1d finished 00:10  Fri session filed as SATURDAY

    -- costing one capture outright and mis-dating three more, in the one
    dataset whose whole value is that it accumulates correctly over time.

    The last bar is read from the OPEN trades: run_backtest marks a
    position still open at the end of the data to market at that final
    bar, so any is_open row's exit_date IS the last bar's timestamp. There
    are ~1,900 of them per interval, so this is not a thin inference.

    Falling back in order: open trades, then the latest exit of any trade
    (a floor -- the last bar is at or after it), then the old wall-clock
    behaviour for a database written before is_open existed.
    """
    row = conn.execute(
        """SELECT MAX(t.exit_date)
             FROM trades t
            WHERE t.is_open = 1
              AND t.run_id = (
                  SELECT id FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1
              )""",
        (interval,),
    ).fetchone()
    if row and row[0]:
        return datetime.fromtimestamp(row[0], timezone.utc).date().isoformat()

    # No open positions recorded: either every position closed on the final
    # bar, or this database predates the is_open column. The newest exit is
    # then the best available floor for the last bar.
    row = conn.execute(
        """SELECT MAX(t.exit_date)
             FROM trades t
            WHERE t.run_id = (
                  SELECT id FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1
              )""",
        (interval,),
    ).fetchone()
    if row and row[0]:
        return datetime.fromtimestamp(row[0], timezone.utc).date().isoformat()

    row = conn.execute(
        "SELECT run_at FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1", (interval,)
    ).fetchone()
    return row[0][:10] if row else None


def _closes(conn, interval: str) -> dict[str, float]:
    """Close per ticker from the snapshot's technicals -- the same price
    the grade was computed against, so the forward return in grade_report
    is anchored to exactly what was scored."""
    technicals = db_module.all_technicals(conn, interval)
    if technicals.empty:
        return {}
    rows = technicals.dropna(subset=["close"])
    return dict(zip(rows["ticker"], rows["close"].astype(float)))


def capture(
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    history_path: Path = grade_history.DEFAULT_PATH,
    as_of: str | None = None,
    min_trades: int = 15,
) -> tuple[str | None, int, int]:
    """Score the universe from the latest snapshot and append it.

    Returns (as_of, rows_appended, rows_in_history). rows_appended is 0
    when there is nothing to grade -- a database with no run for this
    interval yet, which is not an error on a first run.
    """
    conn = db_module.connect(db_path)
    as_of = as_of or _run_date(conn, interval)
    closes = _closes(conn, interval)
    conn.close()
    if as_of is None:
        return None, 0, 0

    scores = compute_universe_setup_scores(db_path=db_path, interval=interval, min_trades=min_trades)
    if scores.empty:
        return as_of, 0, 0

    records = grade_history.to_records(scores, as_of, interval, closes=closes, valuation_available=True)
    total = grade_history.append(records, history_path)
    return as_of, len(records), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--history-file", default=str(grade_history.DEFAULT_PATH))
    parser.add_argument("--as-of", default=None, help="override the date these grades are filed under")
    parser.add_argument("--min-trades", type=int, default=15)
    args = parser.parse_args()

    as_of, appended, total = capture(
        db_path=args.db_file,
        interval=args.interval,
        history_path=Path(args.history_file),
        as_of=args.as_of,
        min_trades=args.min_trades,
    )
    if as_of is None:
        print(f"no {args.interval} run in {args.db_file} yet, nothing to capture", file=sys.stderr)
        return
    if not appended:
        print(f"{as_of}: no tickers scored for {args.interval}, nothing appended", file=sys.stderr)
        return

    history = grade_history.load(Path(args.history_file), interval=args.interval)
    today = history[history["as_of"] == as_of]
    counts = today["grade"].value_counts()
    summary = " ".join(f"{g}:{int(counts.get(g, 0))}" for g in "ABCD")
    print(f"{as_of} ({args.interval}): captured {appended} grades  {summary}; history holds {total} rows")


if __name__ == "__main__":
    main()
