"""Walk-forward validation: does the "best strategy per regime" pick made
on one time window still win when applied to the window right after it --
data the pick never saw?

Every number the dashboard/recommend_cli show is in-sample: the same
trade history that crowns a strategy the "best fit" for a regime is the
history being used to grade it, so part of what looks like edge can just
be curve-fit noise, especially in the thinner buckets. Walk-forward is the
standard fix -- split history into sequential folds, pick the winner using
only data up to the fold boundary, then measure that same pick's
performance strictly after the boundary.

This reuses the trade-level rows already in the SQLite store instead of
re-running the backtest per fold. That's valid, not a shortcut: every
signal in strategies.py and every regime feature in regime.py is causal
(rolling/ewm windows only look backward), so a trade computed once over
the full history is identical to what the same strategy would have
produced live at that trade's entry date. Re-bucketing those trades by
entry_date into folds is equivalent to re-running the backtest on each
truncated window, without paying to actually do that.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import db as db_module


def _trades_frame(conn, interval: str, run_at: str | None = None) -> pd.DataFrame:
    run_at = run_at or db_module._latest_run(conn, interval)
    if run_at is None:
        return pd.DataFrame()
    return pd.read_sql_query(
        """SELECT t.ticker, s.name AS strategy, t.regime, t.direction,
                  t.entry_date, t.return_pct
           FROM trades t
           JOIN runs r ON r.id = t.run_id
           JOIN strategies s ON s.id = t.strategy_id
           WHERE r.run_at = ? AND r.interval = ?""",
        conn,
        params=(run_at, interval),
    )


def _pick_best(window: pd.DataFrame, min_trades: int) -> dict[tuple[str, str], str]:
    """Best strategy per (regime, direction) bucket within this window,
    by average return per trade -- same selection rule best_strategy()
    uses at query time, just scoped to one fold instead of the full
    history."""
    picks: dict[tuple[str, str], str] = {}
    for (regime, direction), group in window.groupby(["regime", "direction"]):
        agg = group.groupby("strategy")["return_pct"].agg(trade_count="count", avg_return_pct="mean")
        agg = agg[agg["trade_count"] >= min_trades]
        if agg.empty:
            continue
        picks[(regime, direction)] = agg["avg_return_pct"].idxmax()
    return picks


def run_walk_forward(
    conn,
    interval: str,
    n_folds: int = 4,
    min_trades: int = 10,
    run_at: str | None = None,
) -> pd.DataFrame:
    """Anchored walk-forward over n_folds sequential test windows.

    History is cut into n_folds + 1 equal-width calendar buckets. Fold i
    trains on every trade before bucket i+1's start (an expanding window,
    not a fixed-size one -- each fold gets to use strictly more history
    than the last, mirroring how the real system accumulates data over
    time) and tests on bucket i+1 alone, which the pick never saw.

    Returns one row per (fold, regime, direction) with the in-sample pick
    and how that exact pick performed out-of-sample. A bucket with no
    strategy clearing min_trades in-sample is skipped for that fold.
    """
    trades = _trades_frame(conn, interval, run_at)
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["entry_dt"] = pd.to_datetime(trades["entry_date"], unit="s", utc=True)
    trades = trades.sort_values("entry_dt")

    edges = pd.date_range(trades["entry_dt"].min(), trades["entry_dt"].max(), periods=n_folds + 2)

    rows = []
    for i in range(n_folds):
        train = trades[trades["entry_dt"] < edges[i + 1]]
        test = trades[(trades["entry_dt"] >= edges[i + 1]) & (trades["entry_dt"] < edges[i + 2])]
        picks = _pick_best(train, min_trades)

        for (regime, direction), strategy in picks.items():
            train_bucket = train[
                (train["regime"] == regime) & (train["direction"] == direction) & (train["strategy"] == strategy)
            ]
            test_bucket = test[
                (test["regime"] == regime) & (test["direction"] == direction) & (test["strategy"] == strategy)
            ]
            rows.append(
                {
                    "fold": i,
                    "train_through": edges[i + 1],
                    "test_through": edges[i + 2],
                    "regime": regime,
                    "direction": direction,
                    "strategy": strategy,
                    "train_trades": len(train_bucket),
                    "train_avg_return_pct": train_bucket["return_pct"].mean(),
                    "test_trades": len(test_bucket),
                    "test_avg_return_pct": test_bucket["return_pct"].mean() if len(test_bucket) else None,
                    "test_win_rate": (test_bucket["return_pct"] > 0).mean() * 100 if len(test_bucket) else None,
                }
            )

    return pd.DataFrame(rows)


def summarize_walk_forward(detail: pd.DataFrame) -> pd.DataFrame:
    """Roll the per-fold detail up to one row per (regime, direction,
    strategy): how many folds picked it, and how it did out-of-sample
    across those folds pooled together. `oos_win_folds` -- the fraction of
    folds where the pick's out-of-sample average return was positive --
    is the headline stability number: a strategy that wins most folds
    in-sample but loses money out-of-sample most of the time is exactly
    the overfitting this whole exercise is meant to catch."""
    if detail.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> pd.Series:
        tested = group.dropna(subset=["test_avg_return_pct"])
        return pd.Series(
            {
                "folds_picked": len(group),
                "folds_with_oos_trades": len(tested),
                "oos_trade_count": int(tested["test_trades"].sum()),
                "oos_avg_return_pct": tested["test_avg_return_pct"].mean() if not tested.empty else None,
                "oos_win_rate": tested["test_win_rate"].mean() if not tested.empty else None,
                "oos_positive_folds": (tested["test_avg_return_pct"] > 0).mean() * 100 if not tested.empty else None,
            }
        )

    summary = detail.groupby(["regime", "direction", "strategy"]).apply(_agg)
    return summary.reset_index().sort_values(["regime", "direction", "oos_avg_return_pct"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validate the regime/strategy picks against out-of-sample folds.")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--min-trades", type=int, default=10, help="minimum in-sample trades for a fold to trust a pick")
    args = parser.parse_args()

    conn = db_module.connect(args.db_file)
    detail = run_walk_forward(conn, args.interval, n_folds=args.n_folds, min_trades=args.min_trades)
    conn.close()

    if detail.empty:
        print("No trades available for this interval.")
        return

    summary = summarize_walk_forward(detail)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
