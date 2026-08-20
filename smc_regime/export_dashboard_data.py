"""Export the latest SQLite snapshot into the JSON shape the dashboard
Artifact embeds. Re-run this and republish the Artifact whenever the
dashboard needs refreshing -- it is baked at publish time, not live."""
from __future__ import annotations

import argparse
import json

from . import db as db_module

STRATEGY_NAMES = {
    "rsi": "RSI Mean Reversion",
    "bollinger": "Bollinger Mean Reversion",
    "macd": "MACD Crossover",
    "ema_cross": "EMA 20/50 Cross",
    "donchian": "Donchian Breakout",
    "supertrend": "Supertrend",
    "vwap_trend": "VWAP Trend Structure",
    "rsi_macd_reversal": "RSI+MACD Reversal",
    "rsi_macd_trend": "RSI+MACD Trend Continuation",
    "rsi_macd_filter": "RSI-Filtered MACD",
    "rsi_dip_recovery": "RSI Dip & Recovery",
    "rsi_dual_hma": "Dual RSI + Hull MA",
}
MIN_TRADES = 15
REGIME_ORDER = {"choppy": 0, "trending": 1, "parabolic": 2}
DIRECTION_ORDER = {"up": 0, "down": 1, "flat": 2, "n/a": 3}


def _latest_run(conn, interval: str) -> str | None:
    row = conn.execute(
        "SELECT run_at FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1", (interval,)
    ).fetchone()
    return row[0] if row else None


def _interval_payload(conn, interval: str) -> dict:
    run_at = _latest_run(conn, interval)
    if run_at is None:
        return {"run_at": None, "ticker_count": 0, "total_trades": 0, "buckets": []}

    rows = conn.execute(
        """SELECT t.regime, t.direction, s.name,
                  COUNT(*),
                  AVG(CASE WHEN t.return_pct > 0 THEN 1.0 ELSE 0.0 END) * 100,
                  AVG(t.return_pct),
                  SUM(t.return_pct)
           FROM trades t
           JOIN runs r ON r.id = t.run_id
           JOIN strategies s ON s.id = t.strategy_id
           WHERE r.run_at = ? AND r.interval = ?
           GROUP BY t.regime, t.direction, s.name""",
        (run_at, interval),
    ).fetchall()
    ticker_count = conn.execute(
        """SELECT COUNT(DISTINCT t.ticker) FROM trades t
           JOIN runs r ON r.id = t.run_id WHERE r.run_at = ? AND r.interval = ?""",
        (run_at, interval),
    ).fetchone()[0]
    total_trades = conn.execute(
        """SELECT COUNT(*) FROM trades t
           JOIN runs r ON r.id = t.run_id WHERE r.run_at = ? AND r.interval = ?""",
        (run_at, interval),
    ).fetchone()[0]

    buckets: dict[str, list[dict]] = {}
    for regime, direction, strategy, trade_count, win_rate, avg_return_pct, total_return_pct in rows:
        entry = {
            "strategy": strategy,
            "trade_count": trade_count,
            "win_rate": round(win_rate, 1),
            "avg_return_pct": round(avg_return_pct, 2),
            "total_return_pct": round(total_return_pct, 1),
            "thin": trade_count < MIN_TRADES,
        }
        buckets.setdefault(f"{regime}|{direction}", []).append(entry)

    bucket_list = []
    for key, entries in buckets.items():
        regime, direction = key.split("|")
        entries.sort(key=lambda e: e["avg_return_pct"], reverse=True)
        qualified = [e for e in entries if not e["thin"]]
        best = max(qualified, key=lambda e: e["avg_return_pct"]) if qualified else None
        bucket_list.append({"regime": regime, "direction": direction, "best": best, "all": entries})

    bucket_list.sort(key=lambda b: (REGIME_ORDER[b["regime"]], DIRECTION_ORDER[b["direction"]]))
    return {"run_at": run_at, "ticker_count": ticker_count, "total_trades": total_trades, "buckets": bucket_list}


def export(db_path: str) -> dict:
    conn = db_module.connect(db_path)
    out = {
        "strategy_names": STRATEGY_NAMES,
        "min_trades": MIN_TRADES,
        "1d": _interval_payload(conn, "1d"),
        "1h": _interval_payload(conn, "1h"),
        "sectors": [
            {"sector": s, "count": c}
            for s, c in conn.execute(
                "SELECT sector, COUNT(*) FROM ticker_metadata GROUP BY sector ORDER BY 2 DESC"
            ).fetchall()
        ],
        "exchange_counts": [
            {"exchange": e, "count": c}
            for e, c in conn.execute(
                "SELECT exchange, COUNT(*) FROM ticker_metadata GROUP BY exchange ORDER BY 2 DESC"
            ).fetchall()
        ],
        "ticker_count_total": conn.execute("SELECT COUNT(*) FROM ticker_metadata").fetchone()[0],
    }
    conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the latest snapshot as dashboard JSON.")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--out-file", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    data = export(args.db_file)
    text = json.dumps(data, indent=2)
    if args.out_file:
        with open(args.out_file, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
