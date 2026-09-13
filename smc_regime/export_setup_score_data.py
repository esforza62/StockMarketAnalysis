"""Export a universe-wide A-D setup-grade scorecard as the JSON shape the
Setup Grades dashboard Artifact embeds. Re-run this and republish the
Artifact whenever the dashboard needs refreshing -- it is baked at publish
time, not live, same pattern as export_dashboard_data.py.

Entirely DB-driven (compute_universe_setup_scores(), no live fetching): it
reads whatever the last nightly daily_snapshot run wrote to `latest_regime`
and `trades`, so it only reflects tickers that run has scored.
"""
from __future__ import annotations

import argparse
from collections import Counter

from . import db as db_module
from . import grade_history
from . import jsonfmt
from .setup_score import (
    _ALIGNMENT_MAX,
    _MA_MAX,
    _MACD_MAX,
    _RSI_MAX,
    _SECTOR_INDUSTRY_MAX,
    _STREAK_MAX,
    _VALUATION_MAX,
    _VOLUME_MAX,
    compute_universe_setup_scores,
)

_GRADE_ORDER = ["A", "B", "C", "D"]


def _cell(value, digits: int | None = None):
    """A scored frame's cell as plain JSON.

    pandas turns the None in a mostly-string column into a float NaN when
    it builds the frame, so a ticker with no earnings date arrives here as
    NaN rather than the None score_ticker put in -- and json.dumps writes
    that out as a bare `NaN`, which is not JSON and which JSON.parse
    rejects. Everything optional goes through here on the way out.
    """
    if value is None or value != value:
        return None
    return round(value, digits) if digits is not None else value

# Rendered in this order in the dashboard's expanded row: the four technical
# reads that decide whether the setup is worth taking first, then the
# contextual modifiers.
_COMPONENT_MAXES = [
    ("trend_structure", _MA_MAX),
    ("rsi", _RSI_MAX),
    ("macd", _MACD_MAX),
    ("volume", _VOLUME_MAX),
    ("streak", _STREAK_MAX),
    ("alignment", _ALIGNMENT_MAX),
    ("sector_industry", _SECTOR_INDUSTRY_MAX),
    ("valuation", _VALUATION_MAX),
]


def _latest_run(conn, interval: str) -> str | None:
    row = conn.execute(
        "SELECT run_at FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1", (interval,)
    ).fetchone()
    return row[0] if row else None


def export(db_path: str, interval: str = "1d", min_trades: int = 15) -> dict:
    conn = db_module.connect(db_path)
    run_at = _latest_run(conn, interval)
    # How many tickers the snapshot actually captured technical readings
    # for. The four technical components are the bulk of the score and fall
    # back to half credit without this, so a low count means the grades are
    # mostly neutral filler -- the dashboard warns rather than presenting
    # them as real. Snapshots taken before `technicals` existed have none.
    technicals_covered = len(db_module.all_technicals(conn, interval))
    # News is optional context, so its coverage is reported rather than
    # warned about -- but a run where TIINGO_API_KEY was missing produces
    # zero rows, which on the page looks exactly like a universe with no
    # headlines. The count is what tells those two apart.
    news = db_module.all_news_sentiment(conn)
    news_covered = int((news["article_count"] > 0).sum()) if not news.empty else 0
    # Built while the connection is still open -- the payload is assembled
    # long after conn.close() below, and reading there raised on a closed
    # database rather than silently returning nothing.
    macro_payload = _macro_payload(conn)
    conn.close()

    scores = compute_universe_setup_scores(db_path=db_path, interval=interval, min_trades=min_trades)
    if scores.empty:
        return {"run_at": run_at, "interval": interval, "min_trades": min_trades, "ticker_count": 0, "technicals_covered": technicals_covered, "news_covered": news_covered, "grade_counts": {}, "macro": macro_payload, "sectors": [], "tickers": []}

    # Read once for the whole universe rather than per row: this parses the
    # entire history file, and doing that 415 times would dominate the export.
    streaks = grade_history.grade_streaks(interval=interval)
    grade_counts = Counter(scores["grade"])
    sectors = sorted(scores["sector"].unique().tolist())

    tickers = []
    for _, r in scores.iterrows():
        components = {}
        for name, max_points in _COMPONENT_MAXES:
            detail = (
                f"{r['streak_bars']} bars in current confirmed regime"
                if name == "streak"
                else r[f"{name}_detail"]
            )
            components[name] = {"points": r[f"{name}_points"], "max": max_points, "detail": detail}

        tickers.append(
            {
                "ticker": r["ticker"],
                "sector": r["sector"],
                "industry": r["industry"],
                "regime": r["regime"],
                "direction": r["direction"],
                "grade": r["grade"],
                "total_points": r["total_points"],
                # Null unless the score sits within a point of a grade cut --
                # see setup_score._borderline. Lets the dashboard show that a
                # 75.7 B and a 76.2 A are the same setup wearing different
                # letters, instead of implying a real distinction.
                "borderline": r["borderline"],
                "components": components,
                # Reported, not scored -- which strategy to use once the
                # setup itself is worth taking. See setup_score's docstring.
                "strategy": r["strategy_info"],
                "return_1w": _cell(r["return_1w"], 1),
                "return_1m": _cell(r["return_1m"], 1),
                # Also reported, never scored. Both go out as raw facts --
                # the earnings DATE and the timestamp the news was read --
                # and the page derives "in 6 days" / "3 days stale" against
                # the viewer's own clock. A countdown baked in here would be
                # wrong by however long it has been since the export ran,
                # and wrong in the direction that matters: it would keep
                # claiming an earnings date is ahead after it has passed.
                "news": _cell(r["news"]),
                "volume_ratio": _cell(r["volume_ratio"], 2),
                "volume_pctile": _cell(r["volume_pctile"], 0),
                "earnings_date": _cell(r["earnings_date"]),
                "earnings_is_estimate": _cell(r["earnings_is_estimate"]),
                # How long this ticker has held this grade. The DATE is
                # carried, not a day count -- a stored count is wrong the
                # day after it is written, and these rows outlive the run
                # that made them. `censored` means the streak runs back to
                # the start of the history, so the page must read it as a
                # floor rather than a measurement.
                **_streak_fields(streaks.get(r["ticker"])),
            }
        )

    return {
        "run_at": run_at,
        "interval": interval,
        "min_trades": min_trades,
        "ticker_count": len(tickers),
        "technicals_covered": technicals_covered,
        "news_covered": news_covered,
        "grade_counts": {g: grade_counts.get(g, 0) for g in _GRADE_ORDER},
        "macro": macro_payload,
        "sectors": sectors,
        "tickers": tickers,
    }


def _streak_fields(streak: dict | None) -> dict:
    """Grade-streak fields for one row, all None when the ticker has no
    history yet -- a first run, or a ticker added since capture began."""
    if not streak:
        return {
            "grade_since": None, "grade_observations": None, "grade_censored": None,
            "grade_strict": None, "grade_total": None, "grade_captures": None, "grade_flips": None,
        }
    return {
        "grade_since": streak["since"],
        "grade_observations": streak["observations"],
        "grade_censored": streak["censored"],
        # What the tolerance bought: the unbroken run, the ratio behind the
        # streak, and every disagreement it absorbed. Carried so the page can
        # show a patched streak as patched -- a number that quietly swallowed
        # three reversals would be the false confidence the strict count was
        # free of.
        "grade_strict": streak["strict_observations"],
        "grade_total": streak["total_at_grade"],
        "grade_captures": streak["captures"],
        "grade_flips": streak["flips"],
    }


def _macro_payload(conn) -> dict:
    """Market levels from the last run, plus the committed event schedule.

    The levels carry their OWN fetched_at rather than inheriting the run's:
    the macro step is last in the nightly and can fail on its own, leaving
    yesterday's numbers in place, and a strip that claimed this morning's
    timestamp over last night's prices would be worse than no strip.

    Events are read from the repo at export time rather than the database
    because that is where they live -- there is no point round-tripping a
    committed file through SQLite.
    """
    from . import macro as macro_module

    rows = conn.execute(
        "SELECT symbol, label, unit, price, change_pct, fetched_at FROM macro_levels"
    ).fetchall()
    by_symbol = {r[0]: r for r in rows}
    # Config order, not insertion order: the page reads slowest-moving first.
    # The equity benchmarks are appended last and only ONE of each cash/
    # future pair is ever stored, so both candidates are listed and whichever
    # the last fetch chose is the one that appears. Iterating LEVELS alone
    # silently dropped them when the pairs moved out of that list.
    order = [row[0] for row in macro_module.LEVELS]
    for (cash_symbol, _cl), (fut_symbol, _fl) in macro_module.INDEX_PAIRS:
        order += [cash_symbol, fut_symbol]
    levels = []
    for symbol in order:
        row = by_symbol.get(symbol)
        if row is None:
            continue
        levels.append({
            "symbol": row[0], "label": row[1], "unit": row[2],
            "price": row[3], "change_pct": row[4], "fetched_at": row[5],
        })
    calendar = macro_module.load_calendar()
    return {
        "levels": levels,
        "as_of": max((l["fetched_at"] for l in levels), default=None),
        "events": macro_module.upcoming(calendar),
        "covers_through": calendar.get("covers_through"),
        "expired": macro_module.calendar_expired(calendar),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a universe-wide setup-grade scorecard as dashboard JSON.")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--out-file", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    data = export(args.db_file, args.interval, args.min_trades)
    # One line per ticker: 415 records instead of ~27k lines of punctuation.
    # See jsonfmt for why this is neither indent=2 nor fully compact.
    text = jsonfmt.dumps(data, compact_depth=2)
    if args.out_file:
        with open(args.out_file, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
