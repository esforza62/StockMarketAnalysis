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

CREATE TABLE IF NOT EXISTS latest_regime (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ticker TEXT NOT NULL,
    regime TEXT NOT NULL,
    direction TEXT NOT NULL,
    streak_bars INTEGER NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS valuation (
    ticker TEXT PRIMARY KEY,
    trailing_pe REAL,
    forward_pe REAL,
    fetched_at TEXT
);

-- Same (run_id, ticker) grain as latest_regime and written by the same
-- pass, but kept as its own table so an existing database picks it up via
-- CREATE TABLE IF NOT EXISTS instead of needing an ALTER TABLE migration.
-- Every reading is nullable: short series (a 200-bar MA needs 200 bars) and
-- missing volume are expected, not errors.
CREATE TABLE IF NOT EXISTS technicals (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ticker TEXT NOT NULL,
    close REAL,
    rsi REAL,
    macd_hist REAL,
    macd_hist_prev REAL,
    ma_fast REAL,
    ma_slow REAL,
    volume_ratio REAL,
    price_change_pct REAL,
    PRIMARY KEY (run_id, ticker)
);
"""

_TECHNICAL_COLUMNS = [
    "close", "rsi", "macd_hist", "macd_hist_prev",
    "ma_fast", "ma_slow", "volume_ratio", "price_change_pct",
]


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


def upsert_valuation(conn: sqlite3.Connection, rows: dict[str, dict], fetched_at: str) -> None:
    """`rows` maps ticker -> {trailing_pe, forward_pe} (valuation.fetch_valuation_batch()'s
    shape). A ticker whose fetch failed simply isn't in `rows` and its
    existing row (if any) is left untouched -- a transient fetch failure
    shouldn't erase a previously-known value."""
    if not rows:
        return
    conn.executemany(
        """INSERT INTO valuation (ticker, trailing_pe, forward_pe, fetched_at)
           VALUES (:ticker, :trailing_pe, :forward_pe, :fetched_at)
           ON CONFLICT(ticker) DO UPDATE SET
               trailing_pe = excluded.trailing_pe,
               forward_pe = excluded.forward_pe,
               fetched_at = excluded.fetched_at""",
        [
            {"ticker": ticker.upper(), "trailing_pe": data.get("trailing_pe"), "forward_pe": data.get("forward_pe"), "fetched_at": fetched_at}
            for ticker, data in rows.items()
        ],
    )
    conn.commit()


def get_valuation(conn: sqlite3.Connection, ticker: str) -> dict | None:
    row = conn.execute(
        "SELECT trailing_pe, forward_pe FROM valuation WHERE ticker = ?", (ticker.upper(),)
    ).fetchone()
    if row is None:
        return None
    return {"trailing_pe": row[0], "forward_pe": row[1]}


def all_valuation(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT ticker, trailing_pe, forward_pe FROM valuation", conn)


def write_latest_regime(conn: sqlite3.Connection, run_at: str, interval: str, latest_regime: pd.DataFrame) -> None:
    """Replace any existing latest-regime rows for this (run_at, interval)
    then insert fresh ones -- keeps re-running a snapshot idempotent, same
    pattern as write_trades()."""
    run_id = _get_or_create_run_id(conn, run_at, interval)
    conn.execute("DELETE FROM latest_regime WHERE run_id = ?", (run_id,))
    if latest_regime.empty:
        conn.commit()
        return

    rows = latest_regime.copy()
    rows["run_id"] = run_id
    cols = ["run_id", "ticker", "regime", "direction", "streak_bars"]
    conn.executemany(
        f"INSERT INTO latest_regime ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        rows[cols].itertuples(index=False, name=None),
    )
    conn.commit()


def write_technicals(conn: sqlite3.Connection, run_at: str, interval: str, latest_regime: pd.DataFrame) -> None:
    """Persist the per-ticker technical readings that collect_trades()
    captured alongside the regime, from the same DataFrame. Columns the
    caller didn't supply (an older caller, or a snapshot that predates a
    newly added reading) are written as NULL rather than failing, and the
    scorer treats NULL as neutral."""
    run_id = _get_or_create_run_id(conn, run_at, interval)
    conn.execute("DELETE FROM technicals WHERE run_id = ?", (run_id,))
    if latest_regime.empty:
        conn.commit()
        return

    rows = latest_regime.copy()
    rows["run_id"] = run_id
    for col in _TECHNICAL_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    cols = ["run_id", "ticker", *_TECHNICAL_COLUMNS]
    conn.executemany(
        f"INSERT INTO technicals ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
        rows[cols].astype(object).where(pd.notna(rows[cols]), None).itertuples(index=False, name=None),
    )
    conn.commit()


def all_technicals(conn: sqlite3.Connection, interval: str) -> pd.DataFrame:
    """Every tracked ticker's technical readings as of the most recent run
    at this interval -- the technical counterpart to all_latest_regimes()."""
    run_at = _latest_run(conn, interval)
    if run_at is None:
        return pd.DataFrame(columns=["ticker", *_TECHNICAL_COLUMNS])
    return pd.read_sql_query(
        f"""SELECT t.ticker, {', '.join('t.' + c for c in _TECHNICAL_COLUMNS)}
           FROM technicals t
           JOIN runs r ON r.id = t.run_id
           WHERE r.run_at = ? AND r.interval = ?""",
        conn,
        params=(run_at, interval),
    )


def technicals_for_ticker(conn: sqlite3.Connection, interval: str, ticker: str) -> dict | None:
    """The most recently stored technical readings for one ticker at one
    interval, from the last daily_snapshot run that included it."""
    row = conn.execute(
        f"""SELECT {', '.join('t.' + c for c in _TECHNICAL_COLUMNS)}
           FROM technicals t
           JOIN runs r ON r.id = t.run_id
           WHERE r.interval = ? AND t.ticker = ?
           ORDER BY r.run_at DESC LIMIT 1""",
        (interval, ticker.upper()),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(_TECHNICAL_COLUMNS, row))


def latest_regime_for_ticker(conn: sqlite3.Connection, interval: str, ticker: str) -> dict | None:
    """The most recently stored regime/direction/streak_bars for one ticker
    at one interval, from the last daily_snapshot run that included it."""
    row = conn.execute(
        """SELECT lr.regime, lr.direction, lr.streak_bars
           FROM latest_regime lr
           JOIN runs r ON r.id = lr.run_id
           WHERE r.interval = ? AND lr.ticker = ?
           ORDER BY r.run_at DESC LIMIT 1""",
        (interval, ticker.upper()),
    ).fetchone()
    if row is None:
        return None
    return {"regime": row[0], "direction": row[1], "streak_bars": row[2]}


def all_latest_regimes(conn: sqlite3.Connection, interval: str) -> pd.DataFrame:
    """Every tracked ticker's regime/direction/streak_bars plus sector/industry,
    as of the most recent run at this interval -- one query, no live fetching.
    Powers both the sector/industry consensus lookup below and a fully
    DB-driven bulk setup-score pass across the whole tracking universe."""
    run_at = _latest_run(conn, interval)
    if run_at is None:
        return pd.DataFrame(columns=["ticker", "regime", "direction", "streak_bars", "sector", "industry"])
    return pd.read_sql_query(
        """SELECT lr.ticker, lr.regime, lr.direction, lr.streak_bars, m.sector, m.industry
           FROM latest_regime lr
           JOIN runs r ON r.id = lr.run_id
           LEFT JOIN ticker_metadata m ON m.ticker = lr.ticker
           WHERE r.run_at = ? AND r.interval = ?""",
        conn,
        params=(run_at, interval),
    )


def sector_industry_direction_counts(conn: sqlite3.Connection, interval: str = "1d") -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """For the latest `interval` run, counts of each direction ('up',
    'down', 'flat', 'n/a') among all tracked tickers, grouped by sector and
    by industry. Includes every ticker (the caller decrements one vote for
    the ticker being scored, to exclude self-agreement)."""
    regimes = all_latest_regimes(conn, interval)
    sector_counts: dict[str, dict[str, int]] = {}
    industry_counts: dict[str, dict[str, int]] = {}
    for group_col, counts in (("sector", sector_counts), ("industry", industry_counts)):
        for group_value, sub in regimes.groupby(group_col):
            counts[group_value] = sub["direction"].value_counts().to_dict()
    return sector_counts, industry_counts


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
    industry: str | None = None,
    sector: str | None = None,
    min_trades: int = 15,
) -> dict | None:
    """Symbol-specific -> industry-level -> sector-level -> full-population
    pooled fallback. `industry` is the GICS Industry Group (25 buckets,
    finer than the 11 GICS sectors) stored in ticker_metadata.industry.

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
    if industry is not None:
        industry_tickers = pd.read_sql_query(
            "SELECT ticker FROM ticker_metadata WHERE industry = ?", conn, params=(industry,)
        )["ticker"]
        tiers.append(("industry", base[base["ticker"].isin(industry_tickers)]))
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
