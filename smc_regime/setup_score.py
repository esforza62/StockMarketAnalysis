"""Composite A/B/C/D setup score: how strong and reliable is the
historically-best-matching strategy's edge for a ticker's CURRENT regime,
right now.

Deliberately NOT a directional "is this bullish" score -- trending down is
graded the same as trending up when the edge is just as strong. This system
trades dips within downtrends via rsi_dip_recovery (one of the best-sampled,
best-performing buckets in this whole project), so a downtrend is not
inherently worse than an uptrend; grading it that way would undervalue
exactly the setups this system is built to find.

News sentiment is reported alongside the grade as context, not folded into
the score: negative news during a dip-buy setup is often the reason the dip
happened in the first place, not evidence the setup is bad. Mechanically
penalizing it would undermine the same trades above.

Score components (100 points total):
  - Edge strength & sample quality (50 pts) -- from db.best_strategy()'s
    avg_return_pct / win_rate, scaled down by which fallback tier
    (symbol/industry/sector/pooled) it came from. Zero if no strategy
    clears min_trades anywhere.
  - Regime confirmation stability (25 pts) -- how many consecutive bars
    the CURRENT confirmed regime/direction has held. A regime that just
    barely cleared the confirmation threshold is a much weaker signal
    than one that's held for a while; full credit at _STREAK_FULL_CREDIT_BARS
    bars, scaled linearly below that.
  - Multi-timeframe alignment (25 pts) -- does the daily direction agree
    with the weekly direction? Full credit if they agree, half credit if
    one side is choppy (no active disagreement), zero if they actively
    oppose.

Grade thresholds (A >= 75, B >= 55, C >= 35, D < 35) and the reference
points above are initial, documented defaults -- tune them once this has
been checked against real tickers, not treated as fixed.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import db as db_module
from .data import fetch_ohlcv
from .metadata import get_industry, get_sector
from .news import ticker_sentiment
from .regime import classify_regime, confirmed_regime

_TIER_WEIGHT = {"symbol": 1.0, "industry": 0.8, "sector": 0.6, "pooled": 0.4}
_STREAK_FULL_CREDIT_BARS = 15
_GRADE_THRESHOLDS = [(75, "A"), (55, "B"), (35, "C"), (0, "D")]


def regime_streak_bars(regime_df: pd.DataFrame) -> int:
    """How many consecutive bars, ending at the last bar, share the same
    (regime, direction) pair as the last bar."""
    key = regime_df["regime"] + "|" + regime_df["direction"]
    last = key.iloc[-1]
    streak = 0
    for value in key.iloc[::-1]:
        if value != last:
            break
        streak += 1
    return streak


def _edge_points(best: dict | None) -> tuple[float, str]:
    if best is None:
        return 0.0, "no strategy clears the trade-count threshold in this regime bucket"
    return_component = min(max(best["avg_return_pct"], 0.0), 20.0) / 20.0
    win_rate_component = best["win_rate"] / 100.0
    raw = return_component * 0.6 + win_rate_component * 0.4
    tier_weight = _TIER_WEIGHT[best["source"]]
    points = raw * 50 * tier_weight
    detail = (
        f"{best['strategy']} @ {best['source']} tier, {best['trade_count']} trades, "
        f"{best['win_rate']:.0f}% win rate, {best['avg_return_pct']:+.2f}%/trade avg"
    )
    return points, detail


def _streak_points(streak_bars: int) -> float:
    return min(streak_bars / _STREAK_FULL_CREDIT_BARS, 1.0) * 25


def _alignment_points(daily_regime: dict, weekly_regime: dict) -> tuple[float, str]:
    d_regime, d_dir = daily_regime["regime"], daily_regime["direction"]
    w_regime, w_dir = weekly_regime["regime"], weekly_regime["direction"]

    if d_dir == w_dir and d_dir not in ("n/a", "flat"):
        return 25.0, f"daily and weekly both {d_dir}"
    if d_regime == "choppy" or w_regime == "choppy":
        return 12.5, f"no conflict, but one side is choppy (daily={d_regime}/{d_dir}, weekly={w_regime}/{w_dir})"
    return 0.0, f"daily and weekly actively disagree (daily={d_regime}/{d_dir}, weekly={w_regime}/{w_dir})"


def _grade(total_points: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if total_points >= threshold:
            return grade
    return "D"


def compute_setup_score(
    ticker: str,
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    min_trades: int = 15,
    confirm_bars: int = 3,
    news_days: int = 7,
) -> dict:
    """Fetch live data (Tiingo), classify the current regime, look up the
    best-matching historical strategy, and combine into an A-D setup grade.
    Also fetches daily+weekly for alignment (reusing the primary fetch if
    `interval` is already one of those) and recent news sentiment as
    separate context, not part of the score."""
    df = fetch_ohlcv(ticker, period="2y", interval=interval)
    regime_df = confirmed_regime(classify_regime(df), confirm_bars=confirm_bars)
    current = regime_df.iloc[-1]
    streak_bars = regime_streak_bars(regime_df)

    if interval == "1d":
        daily_df = regime_df
    else:
        daily_df = confirmed_regime(classify_regime(fetch_ohlcv(ticker, period="2y", interval="1d")), confirm_bars=confirm_bars)
    if interval == "1w":
        weekly_df = regime_df
    else:
        weekly_df = confirmed_regime(classify_regime(fetch_ohlcv(ticker, period="5y", interval="1w")), confirm_bars=confirm_bars)

    conn = db_module.connect(db_path)
    sector = get_sector(ticker)
    industry = get_industry(ticker)
    best = db_module.best_strategy(
        conn, interval, current["regime"], current["direction"],
        ticker=ticker, industry=industry, sector=sector, min_trades=min_trades,
    )
    conn.close()

    edge_pts, edge_detail = _edge_points(best)
    streak_pts = _streak_points(streak_bars)
    align_pts, align_detail = _alignment_points(daily_df.iloc[-1], weekly_df.iloc[-1])
    total = edge_pts + streak_pts + align_pts

    news = None
    if news_days > 0:
        try:
            news = ticker_sentiment(ticker, days=news_days)
        except Exception:
            news = None

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "regime": current["regime"],
        "direction": current["direction"],
        "grade": _grade(total),
        "total_points": round(total, 1),
        "components": {
            "edge": {"points": round(edge_pts, 1), "max": 50, "detail": edge_detail},
            "streak": {"points": round(streak_pts, 1), "max": 25, "detail": f"{streak_bars} bars in current confirmed regime"},
            "alignment": {"points": round(align_pts, 1), "max": 25, "detail": align_detail},
        },
        "news": news,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute an A-D setup score for a ticker's current regime.")
    parser.add_argument("ticker")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--confirm-bars", type=int, default=3)
    parser.add_argument("--news-days", type=int, default=7)
    args = parser.parse_args()

    result = compute_setup_score(
        args.ticker, args.db_file, args.interval, args.min_trades, args.confirm_bars, args.news_days
    )
    print(f"{result['ticker']} ({result['interval']}): {result['regime']}/{result['direction']} -> grade {result['grade']} ({result['total_points']}/100)")
    for name, c in result["components"].items():
        print(f"  {name:>10}: {c['points']:>5.1f}/{c['max']}  {c['detail']}")
    if result["news"] and result["news"]["article_count"] > 0:
        n = result["news"]
        print(f"  news: {n['article_count']} articles, {n['label']} (avg compound {n['avg_compound']:+.3f}), counts: {n['counts']}")
    elif result["news"] is not None:
        print("  news: no recent articles")


if __name__ == "__main__":
    main()
