"""Grade the tracking universe as of past dates, from price history alone.

The point is to be able to ask "what did this score on 1 September, and how
have those names done since" without waiting months for nightly capture to
accumulate. Grades were never persisted before (see grade_history), so the
only way to get a September answer in September is to reconstruct it.

What is faithful and what is not
--------------------------------
90 of the 100 grade points are functions of price and volume history, so
they reconstruct exactly: trend structure (20), RSI (15), MACD (15), volume
(15), regime streak (10), daily/weekly alignment (10), sector/industry peer
agreement (5). Every one is computed here through `setup_score.score_ticker`
-- the same function the live path calls -- so a backfilled grade cannot
drift from a live one by way of a second copy of the weighting.

Valuation (10) does NOT reconstruct. Yahoo serves only current trailing and
forward P/E; there is no as-of-date history behind it. Those 10 points are
held at their neutral half credit, exactly as they are for a ticker with no
valuation row, and every row written is flagged `valuation_available:
false` so a report can never compare a backfilled grade against a live one
as though both were measured the same way.

Two consequences worth stating plainly, since they bound what the backfill
can prove:

* A backfilled score is pulled toward the middle by up to 5 points against
  its live counterpart, which is enough to move a name across a grade cut
  (the A/B line is 76). Grade *counts* from a backfill are therefore not
  comparable to live ones; the ordering within a date is what holds up.
* Peer agreement is computed from the tickers in this run's universe, which
  is today's universe. A name added to tracking last week still votes on
  its sector for dates before it was added.

Lookahead
---------
Every input is taken from bars at or before the as-of date. The regime
pipeline is causal -- classifying the full series and slicing the result
gives bar-for-bar the same answer as classifying the truncated series --
which is what lets this classify once per ticker instead of once per
(ticker, date). Weekly bars are the exception: resampling the full daily
series would build the as-of week's bar out of days that had not happened
yet, so the weekly frame is rebuilt from the truncated daily series for
each date.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from . import db as db_module
from . import grade_history
from .data import _resample_weekly, fetch_ohlcv
from .regime import classify_regime, confirmed_regime, regime_streak_bars
from .setup_score import score_ticker
from .technicals import technical_snapshot

# The 200-bar MA is the longest lookback any component needs; the regime
# features want a couple hundred bars of their own to settle. Two extra
# calendar years before the earliest as-of date covers both with margin,
# and the fetch is cached, so over-fetching costs one request per ticker.
_WARMUP_DAYS = 730

_BACKFILL_VALUATION_DETAIL = (
    "valuation not reconstructible for a past date (only current P/E is published); "
    "held at neutral half credit"
)


def _cache_path(cache_dir: Path, ticker: str, start_date: str) -> Path:
    return cache_dir / f"{ticker.upper()}_{start_date}.pkl"


def load_bars(ticker: str, start_date: str, cache_dir: Path | None) -> pd.DataFrame:
    """Daily bars for `ticker` from `start_date`, cached on disk.

    The cache is what makes this re-runnable: adding a date to the range or
    fixing a scoring bug should not mean re-fetching 400+ tickers.
    """
    if cache_dir is None:
        return fetch_ohlcv(ticker, interval="1d", start_date=start_date)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker, start_date)
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    df = fetch_ohlcv(ticker, interval="1d", start_date=start_date)
    with path.open("wb") as f:
        pickle.dump(df, f)
    return df


def universe(db_path: str) -> pd.DataFrame:
    """Tickers with their sector and industry, from `ticker_metadata`.

    Read from metadata rather than `latest_regime` deliberately: that table
    is keyed on `run_id`, and run ids restart at 1 every rebuild, so it
    describes whichever run happens to be in the database right now.
    """
    conn = db_module.connect(db_path)
    rows = pd.read_sql_query(
        "SELECT ticker, sector, industry FROM ticker_metadata ORDER BY ticker", conn
    )
    conn.close()
    rows["sector"] = rows["sector"].fillna("Unknown").replace("", "Unknown")
    rows["industry"] = rows["industry"].fillna("Unknown").replace("", "Unknown")
    return rows


def _weekly_state(daily_slice: pd.DataFrame, confirm_bars: int):
    """Weekly regime row as of the end of `daily_slice`.

    Rebuilt per date rather than sliced out of one full-series resample:
    the bar covering the as-of date is labelled with that week's Friday and
    would otherwise be aggregated from days after the as-of date.
    """
    weekly = _resample_weekly(daily_slice)
    if len(weekly) < 2:
        return None, None
    w_regime = confirmed_regime(classify_regime(weekly), confirm_bars=confirm_bars)
    return w_regime, w_regime.iloc[-1]


def _gather(
    ticker_rows: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    interval: str,
    confirm_bars: int,
    min_bars: int,
) -> list[dict]:
    """Everything needed to score every ticker on one date, before peer
    counts are known. Two passes are unavoidable: sector/industry agreement
    is a property of the whole universe on that date, so no ticker can be
    scored until all of them have been classified."""
    gathered = []
    for row in ticker_rows.itertuples():
        daily = bars.get(row.ticker)
        if daily is None:
            continue
        d_slice = daily.loc[:as_of]
        if len(d_slice) < min_bars or d_slice.index[-1] != as_of:
            # Either not enough history to read the indicators, or this
            # ticker did not trade on this date (newly listed, halted,
            # delisted). Grading it would mean carrying a stale bar
            # forward and dating it to today.
            continue

        d_regime_row = regimes[row.ticker].loc[as_of]
        d_streak = regime_streak_bars(regimes[row.ticker].loc[:as_of])
        w_regime, w_row = _weekly_state(d_slice, confirm_bars)

        if interval == "1w":
            if w_regime is None:
                continue
            primary_row = w_row
            primary_streak = regime_streak_bars(w_regime)
            snapshot = technical_snapshot(_resample_weekly(d_slice))
        else:
            primary_row = d_regime_row
            primary_streak = d_streak
            snapshot = technical_snapshot(d_slice)

        gathered.append(
            {
                "ticker": row.ticker,
                "sector": row.sector,
                "industry": row.industry,
                "regime": primary_row["regime"],
                "direction": primary_row["direction"],
                "streak_bars": primary_streak,
                "snapshot": snapshot,
                "daily_row": d_regime_row,
                "weekly_row": w_row,
            }
        )
    return gathered


def _peer_counts(gathered: list[dict]) -> tuple[dict, dict]:
    """Direction counts by sector and by industry, always from the DAILY
    regime -- matching the live path, which reads 1d peer counts even when
    scoring a different interval."""
    sector_counts: dict[str, dict[str, int]] = {}
    industry_counts: dict[str, dict[str, int]] = {}
    for g in gathered:
        direction = g["daily_row"]["direction"]
        for key, counts in ((g["sector"], sector_counts), (g["industry"], industry_counts)):
            counts.setdefault(key, {})
            counts[key][direction] = counts[key].get(direction, 0) + 1
    return sector_counts, industry_counts


def score_date(
    ticker_rows: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    regimes: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    interval: str = "1d",
    confirm_bars: int = 3,
    min_bars: int = 200,
) -> pd.DataFrame:
    """Grade the whole universe as of one date. Empty frame if no ticker
    had a bar on that date (a holiday reached by an explicit --start)."""
    gathered = _gather(ticker_rows, bars, regimes, as_of, interval, confirm_bars, min_bars)
    if not gathered:
        return pd.DataFrame()

    sector_counts, industry_counts = _peer_counts(gathered)
    rows = []
    for g in gathered:
        scored = score_ticker(
            ticker=g["ticker"],
            sector=g["sector"],
            industry=g["industry"],
            regime=g["regime"],
            direction=g["direction"],
            streak_bars=g["streak_bars"],
            snapshot=g["snapshot"],
            daily_row=g["daily_row"],
            weekly_row=g["weekly_row"],
            sector_counts=sector_counts,
            industry_counts=industry_counts,
            valuation=None,
            valuation_missing_detail=_BACKFILL_VALUATION_DETAIL,
            best=None,
        )
        scored["close"] = g["snapshot"].get("close")
        rows.append(scored)
    return pd.DataFrame(rows).sort_values("total_points", ascending=False).reset_index(drop=True)


def trading_dates(bars: dict[str, pd.DataFrame], start: str, end: str) -> list[pd.Timestamp]:
    """Dates the market was actually open, taken from the bars themselves
    rather than a calendar -- no holiday table to fall out of date, and a
    date only survives if real bars exist for it."""
    seen: set[pd.Timestamp] = set()
    for df in bars.values():
        seen.update(df.loc[start:end].index)
    return sorted(seen)


def backfill(
    start: str,
    end: str,
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    history_path: Path = grade_history.DEFAULT_PATH,
    cache_dir: Path | None = None,
    tickers: list[str] | None = None,
    confirm_bars: int = 3,
    min_bars: int = 200,
    progress=None,
) -> tuple[int, int]:
    """Grade every trading day in [start, end] and append to the history.

    Returns (dates_graded, rows_in_history)."""
    ticker_rows = universe(db_path)
    if tickers:
        wanted = {t.upper() for t in tickers}
        ticker_rows = ticker_rows[ticker_rows["ticker"].str.upper().isin(wanted)]

    fetch_from = (pd.Timestamp(start) - pd.Timedelta(days=_WARMUP_DAYS)).strftime("%Y-%m-%d")
    bars: dict[str, pd.DataFrame] = {}
    failed = []
    for row in ticker_rows.itertuples():
        try:
            bars[row.ticker] = load_bars(row.ticker, fetch_from, cache_dir)
        except Exception as exc:  # a delisted or renamed ticker, a bad symbol
            failed.append((row.ticker, str(exc)[:80]))
    if progress:
        progress(f"fetched {len(bars)}/{len(ticker_rows)} tickers" + (f", {len(failed)} failed" if failed else ""))
        for ticker, reason in failed[:10]:
            progress(f"  skipped {ticker}: {reason}")

    # Classify once per ticker over the whole series. Safe because the
    # regime pipeline is causal -- see this module's docstring.
    regimes = {t: confirmed_regime(classify_regime(df), confirm_bars=confirm_bars) for t, df in bars.items()}

    dates = trading_dates(bars, start, end)
    if progress:
        progress(f"{len(dates)} trading day(s) between {start} and {end}")

    graded = 0
    total_rows = 0
    for as_of in dates:
        scores = score_date(ticker_rows, bars, regimes, as_of, interval, confirm_bars, min_bars)
        if scores.empty:
            if progress:
                progress(f"{as_of.date()}: no ticker had a bar, skipped")
            continue
        records = grade_history.to_records(
            scores, as_of.strftime("%Y-%m-%d"), interval, valuation_available=False
        )
        total_rows = grade_history.append(records, history_path)
        graded += 1
        if progress:
            counts = scores["grade"].value_counts()
            summary = " ".join(f"{g}:{int(counts.get(g, 0))}" for g in "ABCD")
            progress(f"{as_of.date()}: {len(scores)} tickers  {summary}")
    return graded, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, help="first as-of date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="last as-of date (default: --start)")
    parser.add_argument("--interval", default="1d", choices=["1d", "1w"])
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--history-file", default=str(grade_history.DEFAULT_PATH))
    parser.add_argument(
        "--cache-dir",
        default=".cache/backfill_bars",
        help="where fetched bars are cached; pass '' to disable caching",
    )
    parser.add_argument("--tickers", default=None, help="comma-separated subset, for a quick check")
    parser.add_argument("--confirm-bars", type=int, default=3)
    parser.add_argument("--min-bars", type=int, default=200)
    args = parser.parse_args()

    graded, rows = backfill(
        start=args.start,
        end=args.end or args.start,
        db_path=args.db_file,
        interval=args.interval,
        history_path=Path(args.history_file),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        tickers=args.tickers.split(",") if args.tickers else None,
        confirm_bars=args.confirm_bars,
        min_bars=args.min_bars,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    print(f"graded {graded} date(s); {args.history_file} now holds {rows} rows")


if __name__ == "__main__":
    main()
