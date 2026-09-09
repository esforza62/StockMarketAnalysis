"""Does higher-timeframe agreement actually improve this pattern's trades?

Tags every sweep_outside entry with what the 1h/4h/1d rungs were saying at
that moment -- using only rungs that had already CLOSED, see
timeframes.align_to_base -- and reports outcomes split by how many agreed.
Reporting rather than gating is the point: a confluence filter added on
faith looks good because it removes trades, and only a split like this
shows whether the ones it removes were actually the losers.

    python -m smc_regime.mtf_cli AAPL NVDA --interval 15m --source yahoo

Yahoo is the default source because it needs no key and serves ~60 days of
15m bars, which is enough to see the signal population but NOT enough to
judge an edge -- a handful of trades per bucket is a description, not a
result. Run it against Tiingo over a longer window before believing any
number it prints.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import patterns as pat
from . import timeframes as tf
from .backtest import run_backtest
from .strategies import sweep_outside_reversal


def _fetch(ticker: str, source: str, interval: str, start_date: str) -> pd.DataFrame:
    if source == "yahoo":
        from .cross_validate import fetch_yahoo_ohlcv

        return fetch_yahoo_ohlcv(ticker, start_date=start_date, interval=interval)
    from .data import fetch_ohlcv

    return fetch_ohlcv(ticker, interval=interval, start_date=start_date)


def agreement_table(df: pd.DataFrame, rules: tuple[str, ...] = tf.LADDER, detector=pat.bar_direction) -> pd.DataFrame:
    """One row per "how many higher rungs were bullish at entry" bucket."""
    signals = sweep_outside_reversal(df)
    ladder = tf.ladder_state(df, rules=rules, detector=detector)
    trades = run_backtest(df, signals)
    if not trades:
        return pd.DataFrame()

    rows = [
        {
            "rungs_agreeing": int(ladder.loc[t.entry_date, "rungs_bullish"]),
            "return_pct": t.return_pct,
        }
        for t in trades
        if t.entry_date in ladder.index
    ]
    tagged = pd.DataFrame(rows)
    return (
        tagged.groupby("rungs_agreeing")["return_pct"]
        .agg(trades="count", win_rate=lambda r: (r > 0).mean() * 100, avg_return_pct="mean")
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--interval", default="15m", help="base interval -- must be the FINEST rung, since the ladder only aggregates upward")
    parser.add_argument("--source", choices=["yahoo", "tiingo"], default="yahoo")
    parser.add_argument("--start-date", default="2026-07-15", help="Yahoo serves roughly 60 days of 15m history; older dates are silently truncated by the vendor")
    parser.add_argument("--detector", choices=["direction", "reversal"], default="direction",
                        help="what a rung agreeing means: which way its last closed bar resolved (dense), or that it printed the same reversal pattern (rare)")
    parser.add_argument("--rungs", default=",".join(tf.LADDER), help='pandas offset aliases, e.g. "60min,240min,1D" -- 240min is the ragged 4h a chart shows, not a true four-hour bar')
    args = parser.parse_args()

    detector = {"direction": pat.bar_direction, "reversal": pat.reversal_signals}[args.detector]

    rules = tuple(r.strip() for r in args.rungs.split(",") if r.strip())

    for ticker in args.tickers:
        try:
            df = _fetch(ticker, args.source, args.interval, args.start_date)
        except Exception as exc:
            print(f"{ticker}: fetch failed -- {exc}")
            continue

        print(f"\n{ticker}  {len(df)} {args.interval} bars  rungs={','.join(rules)}")
        for rule in rules:
            htf = tf.session_resample(df, rule)
            per_session = htf.groupby(tf._session_dates(htf.index)).size()
            print(
                f"  {rule:>7}: {len(htf):>5} bars, {per_session.min()}-{per_session.max()} per session, "
                f"source bars/bar {int(htf['source_bars'].min())}-{int(htf['source_bars'].max())}"
            )

        table = agreement_table(df, rules, detector)
        if table.empty:
            print("  no trades")
            continue
        print(f"  {'rungs agreeing':>15} {'trades':>7} {'win %':>7} {'avg %':>8}")
        for _, row in table.iterrows():
            print(
                f"  {int(row['rungs_agreeing']):>15} {int(row['trades']):>7} "
                f"{row['win_rate']:>7.0f} {row['avg_return_pct']:>8.2f}"
            )


if __name__ == "__main__":
    main()
