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
from .setup_score import compute_universe_setup_scores

_GRADE_ORDER = ["A", "B", "C", "D"]


def _latest_run(conn, interval: str) -> str | None:
    row = conn.execute(
        "SELECT run_at FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1", (interval,)
    ).fetchone()
    return row[0] if row else None


def export(db_path: str, interval: str = "1d", min_trades: int = 15) -> dict:
    conn = db_module.connect(db_path)
    run_at = _latest_run(conn, interval)
    conn.close()

    scores = compute_universe_setup_scores(db_path=db_path, interval=interval, min_trades=min_trades)
    if scores.empty:
        return {"run_at": run_at, "interval": interval, "min_trades": min_trades, "ticker_count": 0, "grade_counts": {}, "sectors": [], "tickers": []}

    grade_counts = Counter(scores["grade"])
    sectors = sorted(scores["sector"].unique().tolist())

    tickers = []
    for _, r in scores.iterrows():
        tickers.append(
            {
                "ticker": r["ticker"],
                "sector": r["sector"],
                "industry": r["industry"],
                "regime": r["regime"],
                "direction": r["direction"],
                "grade": r["grade"],
                "total_points": r["total_points"],
                "components": {
                    "edge": {"points": r["edge_points"], "max": 40, "detail": r["edge_detail"]},
                    "streak": {"points": r["streak_points"], "max": 20, "detail": f"{r['streak_bars']} bars in current confirmed regime"},
                    "alignment": {"points": r["alignment_points"], "max": 20, "detail": r["alignment_detail"]},
                    "sector_industry": {"points": r["sector_industry_points"], "max": 20, "detail": r["sector_industry_detail"]},
                },
            }
        )

    return {
        "run_at": run_at,
        "interval": interval,
        "min_trades": min_trades,
        "ticker_count": len(tickers),
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
