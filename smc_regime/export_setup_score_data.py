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
import json
from collections import Counter

from . import db as db_module
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
    conn.close()

    scores = compute_universe_setup_scores(db_path=db_path, interval=interval, min_trades=min_trades)
    if scores.empty:
        return {"run_at": run_at, "interval": interval, "min_trades": min_trades, "ticker_count": 0, "technicals_covered": technicals_covered, "grade_counts": {}, "sectors": [], "tickers": []}

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
                "components": components,
                # Reported, not scored -- which strategy to use once the
                # setup itself is worth taking. See setup_score's docstring.
                "strategy": r["strategy_info"],
            }
        )

    return {
        "run_at": run_at,
        "interval": interval,
        "min_trades": min_trades,
        "ticker_count": len(tickers),
        "technicals_covered": technicals_covered,
        "grade_counts": {g: grade_counts.get(g, 0) for g in _GRADE_ORDER},
        "sectors": sectors,
        "tickers": tickers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a universe-wide setup-grade scorecard as dashboard JSON.")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--out-file", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    data = export(args.db_file, args.interval, args.min_trades)
    text = json.dumps(data, indent=2)
    if args.out_file:
        with open(args.out_file, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
