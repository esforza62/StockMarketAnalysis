"""SQLite store for regime-tagged trades and ticker metadata.

Trade-level rows (not pre-pooled) are stored so that pooling happens at
query time -- this is what makes the fallback hierarchy possible:
symbol-specific stats (when a ticker has enough of its own trades) ->
sector-level stats (pooling every ticker that shares a GICS sector) ->
full-population pooled stats (final fallback). A vector database was
considered and rejected for this -- these are exact structured lookups
(ticker, regime, direction, strategy), not similarity search.

`run_at`/`interval` and `strategy` are normalized into small lookup
tables rather than repeated as text on every trade row -- with 500k+
rows and only a handful of distinct runs/strategies, that repetition was
most of the file's size. `entry_date`/`exit_date` are stored as Unix
epoch seconds (INTEGER) instead of ISO text for the same reason.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent.parent / "backtest_logs" / "smc_regime.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_at TEXT NOT NULL,
    interval TEXT NOT NULL,
    UNIQUE (run_at, interval)
);

CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS trades (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ticker TEXT NOT NULL,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    regime TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_date INTEGER NOT NULL,
    exit_date INTEGER NOT NULL,
    return_pct REAL NOT NULL,
    win INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_lookup
    ON trades (run_id, regime, direction, strategy_id, ticker);

CREATE TABLE IF NOT EXISTS ticker_metadata (
    ticker TEXT PRIMARY KEY,
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    last_updated TEXT
);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def clear_trades(conn: sqlite3.Connection) -> None:
    """Drop all trade rows (and the run records they belonged to) -- use
    before a full rebuild (e.g. a new start date or a changed tracking
    universe), not for routine daily snapshots. `strategies` is left
    alone since strategy names are stable across rebuilds."""
    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM runs")
    conn.commit()


def _get_or_create_run_id(conn: sqlite3.Connection, run_at: str, interval: str) -> int:
    conn.execute("INSERT OR IGNORE INTO runs (run_at, interval) VALUES (?, ?)", (run_at, interval))
    return conn.execute(
        "SELECT id FROM runs WHERE run_at = ? AND interval = ?", (run_at, interval)
    ).fetchone()[0]


def _strategy_ids(conn: sqlite3.Connection, names: list[str]) -> dict[str, int]:
    unique_names = list(set(names))
    conn.executemany("INSERT OR IGNORE INTO strategies (name) VALUES (?)", [(n,) for n in unique_names])
    rows = conn.execute(
        f"SELECT id, name FROM strategies WHERE name IN ({', '.join(['?'] * len(unique_names))})", unique_names
    ).fetchall()
    return {name: id_ for id_, name in rows}


def write_trades(conn: sqlite3.Connection, run_at: str, interval: str, trades: pd.DataFrame) -> None:
    """Replace any existing rows for this (run_at, interval) then insert
    fresh trade-level rows -- keeps re-running a snapshot idempotent."""
    run_id = _get_or_create_run_id(conn, run_at, interval)
    conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
    if trades.empty:
        conn.commit()
        return

    strategy_ids = _strategy_ids(conn, trades["strategy"].tolist())

    rows = trades.copy()
    rows["run_id"] = run_id
    rows["strategy_id"] = rows["strategy"].map(strategy_ids)
    # Not .astype("int64") // 10**9: that assumes nanosecond-resolution
    # datetime64, but pandas 2.0+ can produce other resolutions (observed:
    # datetime64[us] from pandas 3.0.5 here), silently under-scaling the
    # result by whatever factor separates the assumed and actual units --
    # every stored date ended up near the 1970 epoch instead of the real
    # trade date. Timedelta floor-division is resolution-agnostic.
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    rows["entry_date"] = (pd.to_datetime(rows["entry_date"], utc=True) - _epoch) // pd.Timedelta(seconds=1)
    rows["exit_date"] = (pd.to_datetime(rows["exit_date"], utc=True) - _epoch) // pd.Timedelta(seconds=1)
    rows["win"] = rows["win"].astype(int)

    cols = ["run_id", "ticker", "strategy_id", "regime", "direction", "entry_date", "exit_date", "return_pct", "win"]
    conn.executemany(
        f"INSERT INTO trades ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        rows[cols].itertuples(index=False, name=None),
    )
    conn.commit()


def upsert_ticker_metadata(conn: sqlite3.Connection, rows: list[dict[str, str]], last_updated: str) -> None:
    conn.executemany(
        """INSERT INTO ticker_metadata (ticker, exchange, sector, industry, last_updated)
           VALUES (:ticker, :exchange, :sector, :industry, :last_updated)
           ON CONFLICT(ticker) DO UPDATE SET
               exchange = excluded.exchange,
               sector = excluded.sector,
               industry = excluded.industry,
               last_updated = excluded.last_updated""",
        [{**row, "industry": row.get("industry"), "last_updated": last_updated} for row in rows],
    )
    conn.commit()


def _latest_run(conn: sqlite3.Connection, interval: str) -> str | None:
    row = conn.execute(
        "SELECT run_at FROM runs WHERE interval = ? ORDER BY run_at DESC LIMIT 1", (interval,)
    ).fetchone()
    return row[0] if row else None


def _aggregate(rows: pd.DataFrame) -> dict | None:
    if rows.empty:
        return None
    returns = rows["return_pct"]
    return {
        "trade_count": int(len(rows)),
        "win_rate": float((returns > 0).mean() * 100),
        "avg_return_pct": float(returns.mean()),
        "total_return_pct": float(returns.sum()),
    }


def best_strategy(
    conn: sqlite3.Connection,
    interval: str,
    regime: str,
    direction: str,
    ticker: str | None = None,
    sector: str | None = None,
    min_trades: int = 15,
) -> dict | None:
    """Symbol-specific -> sector-level -> full-population pooled fallback.

    Returns the winning strategy's aggregate stats plus which tier
    (`source`) the answer came from, or None if even the pooled tier has
    no data for this (regime, direction).
    """
    run_at = _latest_run(conn, interval)
    if run_at is None:
        return None

    base = pd.read_sql_query(
        """SELECT t.ticker, s.name AS strategy, t.return_pct
           FROM trades t
           JOIN runs r ON r.id = t.run_id
           JOIN strategies s ON s.id = t.strategy_id
           WHERE r.run_at = ? AND r.interval = ? AND t.regime = ? AND t.direction = ?""",
        conn,
        params=(run_at, interval, regime, direction),
    )
    if base.empty:
        return None

    tiers = []
    if ticker is not None:
        tiers.append(("symbol", base[base["ticker"] == ticker.upper()]))
    if sector is not None:
        sector_tickers = pd.read_sql_query(
            "SELECT ticker FROM ticker_metadata WHERE sector = ?", conn, params=(sector,)
        )["ticker"]
        tiers.append(("sector", base[base["ticker"].isin(sector_tickers)]))
    tiers.append(("pooled", base))

    for source, subset in tiers:
        by_strategy = subset.groupby("strategy")
        candidates = []
        for strategy, group in by_strategy:
            agg = _aggregate(group)
            if agg and agg["trade_count"] >= min_trades:
                candidates.append({"strategy": strategy, "source": source, **agg})
        if candidates:
            return max(candidates, key=lambda r: r["avg_return_pct"])

    return None
