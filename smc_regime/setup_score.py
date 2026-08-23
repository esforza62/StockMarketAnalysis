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
  - Edge strength & sample quality (40 pts) -- from db.best_strategy()'s
    avg_return_pct / win_rate, scaled down by which fallback tier
    (symbol/industry/sector/pooled) it came from. Zero if no strategy
    clears min_trades anywhere.
  - Regime confirmation stability (20 pts) -- how many consecutive bars
    the CURRENT confirmed regime/direction has held. A regime that just
    barely cleared the confirmation threshold is a much weaker signal
    than one that's held for a while; full credit at _STREAK_FULL_CREDIT_BARS
    bars, scaled linearly below that.
  - Multi-timeframe alignment (20 pts) -- does the daily direction agree
    with the weekly direction? Full credit if they agree, half credit if
    one side is choppy (no active disagreement), zero if they actively
    oppose.
  - Sector/industry alignment (20 pts: 10 sector + 10 industry) -- is the
    ticker's daily direction moving WITH the net direction of its peers
    (every other tracked ticker sharing its GICS sector / industry group),
    or against them? Peer directions come from `latest_regime` in the DB,
    a per-ticker snapshot captured during the nightly backtest run (not a
    live fetch of every peer) -- see regime_backtest.collect_trades()'s
    `latest_regime` return value. Full credit per side if peers net the
    same direction, half credit if peers are mixed/no clear majority or
    this ticker itself is flat/choppy (no active disagreement), zero if
    peers net the opposite direction. If no peer data exists yet for a
    sector/industry (DB not yet populated by a nightly run since this
    feature shipped), that side scores half credit rather than zero --
    missing data isn't evidence of misalignment.

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
from .regime import classify_regime, confirmed_regime, regime_streak_bars

_TIER_WEIGHT = {"symbol": 1.0, "industry": 0.8, "sector": 0.6, "pooled": 0.4}
_STREAK_FULL_CREDIT_BARS = 15
_GRADE_THRESHOLDS = [(75, "A"), (55, "B"), (35, "C"), (0, "D")]


def _edge_points(best: dict | None) -> tuple[float, str]:
    if best is None:
        return 0.0, "no strategy clears the trade-count threshold in this regime bucket"
    return_component = min(max(best["avg_return_pct"], 0.0), 20.0) / 20.0
    win_rate_component = best["win_rate"] / 100.0
    raw = return_component * 0.6 + win_rate_component * 0.4
    tier_weight = _TIER_WEIGHT[best["source"]]
    points = raw * 40 * tier_weight
    detail = (
        f"{best['strategy']} @ {best['source']} tier, {best['trade_count']} trades, "
        f"{best['win_rate']:.0f}% win rate, {best['avg_return_pct']:+.2f}%/trade avg"
    )
    return points, detail


def _streak_points(streak_bars: int) -> float:
    return min(streak_bars / _STREAK_FULL_CREDIT_BARS, 1.0) * 20


def _alignment_points(daily_regime: dict, weekly_regime: dict) -> tuple[float, str]:
    d_regime, d_dir = daily_regime["regime"], daily_regime["direction"]
    w_regime, w_dir = weekly_regime["regime"], weekly_regime["direction"]

    if d_dir == w_dir and d_dir not in ("n/a", "flat"):
        return 20.0, f"daily and weekly both {d_dir}"
    if d_regime == "choppy" or w_regime == "choppy":
        return 10.0, f"no conflict, but one side is choppy (daily={d_regime}/{d_dir}, weekly={w_regime}/{w_dir})"
    return 0.0, f"daily and weekly actively disagree (daily={d_regime}/{d_dir}, weekly={w_regime}/{w_dir})"


def _peer_consensus_direction(counts: dict[str, int] | None, self_direction: str) -> str | None:
    """Net direction ('up'/'down'/'mixed') among peers, excluding one vote
    for the ticker being scored itself. None means no peer data at all."""
    if not counts:
        return None
    counts = dict(counts)
    if counts.get(self_direction, 0) > 0:
        counts[self_direction] -= 1
    up, down = counts.get("up", 0), counts.get("down", 0)
    if up == down:
        return "mixed"
    return "up" if up > down else "down"


def _peer_points(self_direction: str, consensus: str | None, label: str) -> tuple[float, str]:
    if consensus is None:
        return 5.0, f"no {label} peer data yet"
    if consensus == "mixed":
        return 5.0, f"{label} peers mixed, no clear majority"
    if self_direction == consensus:
        return 10.0, f"{label} peers also net {consensus}"
    if self_direction in ("flat", "n/a"):
        return 5.0, f"{label} peers net {consensus}, no conflict (this ticker is {self_direction})"
    return 0.0, f"{label} peers net {consensus}, opposes this ticker's {self_direction}"


def _sector_industry_points(
    direction: str,
    sector: str,
    industry: str,
    sector_counts: dict[str, dict[str, int]],
    industry_counts: dict[str, dict[str, int]],
) -> tuple[float, str]:
    sec_pts, sec_detail = _peer_points(direction, _peer_consensus_direction(sector_counts.get(sector), direction), f"sector ({sector})")
    ind_pts, ind_detail = _peer_points(direction, _peer_consensus_direction(industry_counts.get(industry), direction), f"industry ({industry})")
    return sec_pts + ind_pts, f"{sec_detail}; {ind_detail}"


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
    # Peer directions come from the DB (captured during the nightly run),
    # not a live fetch of every other ticker in this ticker's sector/industry
    # -- that would mean dozens of extra live fetches per score computed.
    sector_counts, industry_counts = db_module.sector_industry_direction_counts(conn, interval="1d")
    conn.close()

    edge_pts, edge_detail = _edge_points(best)
    streak_pts = _streak_points(streak_bars)
    align_pts, align_detail = _alignment_points(daily_df.iloc[-1], weekly_df.iloc[-1])
    sec_ind_pts, sec_ind_detail = _sector_industry_points(
        daily_df.iloc[-1]["direction"], sector, industry, sector_counts, industry_counts
    )
    total = edge_pts + streak_pts + align_pts + sec_ind_pts

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
        "sector": sector,
        "industry": industry,
        "grade": _grade(total),
        "total_points": round(total, 1),
        "components": {
            "edge": {"points": round(edge_pts, 1), "max": 40, "detail": edge_detail},
            "streak": {"points": round(streak_pts, 1), "max": 20, "detail": f"{streak_bars} bars in current confirmed regime"},
            "alignment": {"points": round(align_pts, 1), "max": 20, "detail": align_detail},
            "sector_industry": {"points": round(sec_ind_pts, 1), "max": 20, "detail": sec_ind_detail},
        },
        "news": news,
    }


def compute_universe_setup_scores(
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    min_trades: int = 15,
) -> pd.DataFrame:
    """Score every ticker the last nightly snapshot run has a stored regime
    for, entirely from the DB -- no live fetching. This is what powers a
    universe-wide setup-grade dashboard: compute_setup_score() above stays
    live-fetch-based (needed for on-demand lookups of tickers outside the
    tracking universe, and for freshness beyond the last snapshot), but
    scoring 270+ tickers one live fetch at a time is slow and redundant --
    the nightly run already classified every tracked ticker's regime at
    every interval; this just reads that back.

    Alignment falls back to the ticker's own (regime, direction) at half
    credit when daily or weekly snapshot data is missing for it, rather
    than dropping the row -- consistent with how missing peer data is
    treated as neutral, not as evidence against the setup.
    """
    conn = db_module.connect(db_path)
    regimes = db_module.all_latest_regimes(conn, interval)
    if regimes.empty:
        conn.close()
        return pd.DataFrame()

    daily_regimes = regimes if interval == "1d" else db_module.all_latest_regimes(conn, "1d")
    weekly_regimes = regimes if interval == "1w" else db_module.all_latest_regimes(conn, "1w")
    daily_by_ticker = daily_regimes.set_index("ticker") if not daily_regimes.empty else None
    weekly_by_ticker = weekly_regimes.set_index("ticker") if not weekly_regimes.empty else None
    sector_counts, industry_counts = db_module.sector_industry_direction_counts(conn, interval="1d")

    rows = []
    for _, r in regimes.iterrows():
        ticker, sector, industry = r["ticker"], r["sector"] or "Unknown", r["industry"] or "Unknown"
        best = db_module.best_strategy(
            conn, interval, r["regime"], r["direction"],
            ticker=ticker, industry=industry, sector=sector, min_trades=min_trades,
        )
        edge_pts, edge_detail = _edge_points(best)
        streak_pts = _streak_points(int(r["streak_bars"]))

        d_row = daily_by_ticker.loc[ticker] if daily_by_ticker is not None and ticker in daily_by_ticker.index else None
        w_row = weekly_by_ticker.loc[ticker] if weekly_by_ticker is not None and ticker in weekly_by_ticker.index else None
        if d_row is not None and w_row is not None:
            align_pts, align_detail = _alignment_points(d_row, w_row)
        else:
            align_pts, align_detail = 10.0, "missing daily or weekly snapshot data, assumed neutral"

        daily_direction = d_row["direction"] if d_row is not None else r["direction"]
        sec_ind_pts, sec_ind_detail = _sector_industry_points(daily_direction, sector, industry, sector_counts, industry_counts)

        total = edge_pts + streak_pts + align_pts + sec_ind_pts
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "regime": r["regime"],
                "direction": r["direction"],
                "grade": _grade(total),
                "total_points": round(total, 1),
                "edge_points": round(edge_pts, 1),
                "edge_detail": edge_detail,
                "streak_points": round(streak_pts, 1),
                "streak_bars": int(r["streak_bars"]),
                "alignment_points": round(align_pts, 1),
                "alignment_detail": align_detail,
                "sector_industry_points": round(sec_ind_pts, 1),
                "sector_industry_detail": sec_ind_detail,
            }
        )

    conn.close()
    return pd.DataFrame(rows).sort_values("total_points", ascending=False).reset_index(drop=True)


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
