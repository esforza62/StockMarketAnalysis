"""Composite A/B/C/D setup score: how good does this ticker's setup look
right now, on its own technical merits.

Deliberately NOT a directional "is this bullish" score -- trending down is
graded the same as trending up when the setup is just as clean. This system
trades dips within downtrends via rsi_dip_recovery (one of the best-sampled,
best-performing buckets in this whole project), so a downtrend is not
inherently worse than an uptrend; grading it that way would undervalue
exactly the setups this system is built to find. The technical components
below follow the same rule: an oversold RSI and a pullback to the 50-day are
treated as entry conditions, not as damage.

Strategy edge is NOT scored. Which strategy has historically worked best in
this ticker's regime answers "how would I trade this", which only matters
once the setup itself is worth trading -- so db.best_strategy()'s pick is
reported alongside the grade as information (see the `strategy` key) rather
than folded into it. Scoring it previously let a ticker inherit a high grade
from a strong historical backtest bucket while its own chart looked like
nothing in particular.

News sentiment is likewise reported as context, not scored: negative news
during a dip-buy setup is often the reason the dip happened in the first
place, not evidence the setup is bad.

Score components (100 points total):
  - Trend structure (_MA_MAX pts) -- where price sits against its 50 and
    200 period moving averages, plus whether the 50 is above the 200. Split
    into position (_MA_POSITION_MAX) and MA-vs-MA trend (_MA_CROSS_MAX).
    Price below the 50 but holding the 200 in an uptrend scores well: that
    is a pullback within trend, the setup this system wants, not weakness.
  - RSI (_RSI_MAX pts) -- read against the regime, not on a fixed
    "high = good" scale. In a trend, a pullback into the 30s is the entry
    and a reading above 80 is exhaustion; in a choppy range the logic
    inverts, since buying the top of a range that keeps mean-reverting is
    how you get chopped up.
  - MACD (_MACD_MAX pts) -- the histogram's sign AND whether it is
    expanding or contracting. A positive-but-shrinking histogram is
    momentum fading, and a negative-but-shrinking one is momentum turning
    up; the level alone can't tell those apart.
  - Volume conviction (_VOLUME_MAX pts) -- recent volume against its own
    50-bar baseline, read together with the price move over that same
    window. An advance on heavy volume is conviction; the same advance on
    light volume is not. A decline on light volume (sellers drying up)
    scores better than one on heavy volume (distribution).
  - Regime confirmation stability (_STREAK_MAX pts) -- how many consecutive
    bars the CURRENT confirmed regime/direction has held. A regime that
    just barely cleared the confirmation threshold is a much weaker signal
    than one that's held for a while; full credit at _STREAK_FULL_CREDIT_BARS
    bars, scaled linearly below that.
  - Multi-timeframe alignment (_ALIGNMENT_MAX pts) -- does the daily
    direction agree with the weekly direction? Full credit if they agree,
    half credit if one side is choppy (no active disagreement), zero if
    they actively oppose.
  - Sector/industry alignment (_SECTOR_INDUSTRY_MAX pts, split evenly
    sector/industry) -- is the ticker's daily direction moving WITH the net
    direction of its peers (every other tracked ticker sharing its GICS
    sector / industry group), or against them? Peer directions come from
    `latest_regime` in the DB, a per-ticker snapshot captured during the
    nightly backtest run (not a live fetch of every peer). Full credit per
    side if peers net the same direction, half credit if peers are mixed or
    this ticker itself is flat/choppy, zero if peers net the opposite. No
    peer data scores half credit -- missing data isn't misalignment.
  - Valuation stretch (_VALUATION_MAX pts) -- does the forward P/E sit
    ABOVE the trailing P/E, implying analysts expect earnings to shrink even
    as the market pays today's multiple? This measures RELATIVE divergence
    (forward vs. that same ticker's own trailing P/E), not absolute
    expensiveness. Missing/negative P/E (ETFs, loss-making companies)
    scores half credit -- neutral, since there's nothing to diverge from.

Every technical component falls back to half credit when its input is
missing (short series, no volume data), consistent with how missing peer
and valuation data is treated: absent evidence is neutral, not negative.

Grade thresholds (A >= 76, B >= 68, C >= 57, D < 57), the component weights
and the reference points above are documented defaults -- tune them against
real tickers rather than treating them as fixed.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import db as db_module
from .data import fetch_ohlcv
from .metadata import get_industry, get_sector
from .news import ticker_sentiment
from .regime import classify_regime, confirmed_regime, regime_streak_bars
from .technicals import technical_snapshot
from .valuation import fetch_valuation

_TIER_WEIGHT = {"symbol": 1.0, "industry": 0.8, "sector": 0.6, "pooled": 0.4}
_STREAK_FULL_CREDIT_BARS = 15
# Recalibrated against the first universe-wide run that carried real
# technical readings (2026-08-29, 415 tickers): scores landed 40.1-90.5,
# mean/median 63.9, stdev 9.7. The previous 75/55/35 cuts were set for the
# old edge-weighted model, where a big share of the score came from a
# backtest bucket rather than the chart; against this distribution they put
# 67% of the universe in B and left D empty, so the grade stopped
# discriminating and nothing was ever flagged as "skip". These cuts sit near
# the p90/p65/p25 marks -- roughly A 10% / B 25% / C 40% / D 25% -- which
# keeps A a genuine shortlist and gives D real membership. They describe
# THIS universe's spread, so re-check them if the tracking universe or the
# component weights change materially.
_GRADE_THRESHOLDS = [(76, "A"), (68, "B"), (57, "C"), (0, "D")]

# Scores cluster tightly -- on a typical run the 44 A-grade names span ~16
# points, averaging under 0.4 points between adjacent names. A hard cut
# through that density makes the letter look more decisive than the number
# is: a ticker at 75.7 and one at 76.2 are not meaningfully different setups,
# but they read as B and A. Anything within this margin of a cut is marked
# borderline so the dashboard can say so, rather than silently rounding a
# half-point of noise into a grade change.
_BORDERLINE_MARGIN = 1.0

_MA_MAX = 20
_MA_POSITION_MAX = 12
_MA_CROSS_MAX = 8
_RSI_MAX = 15
_MACD_MAX = 15
_VOLUME_MAX = 15
_STREAK_MAX = 10
_ALIGNMENT_MAX = 10
_SECTOR_INDUSTRY_MAX = 5  # split evenly, _SECTOR_INDUSTRY_MAX / 2 per side
_VALUATION_MAX = 10
_VALUATION_STRETCH_THRESHOLD_PCT = 25.0  # forward P/E this much above trailing -> zero credit

# Volume within +/-15% of its own baseline is "about normal" -- neither
# conviction nor its absence, so it scores the middle of the band either way.
_VOLUME_HIGH_RATIO = 1.15
_VOLUME_LOW_RATIO = 0.85


def _strategy_info(best: dict | None) -> dict:
    """The historically best-matching strategy for this regime bucket, as
    reported context. Not scored -- see the module docstring."""
    if best is None:
        return {"strategy": None, "detail": "no strategy clears the trade-count threshold in this regime bucket"}
    return {
        "strategy": best["strategy"],
        "source": best["source"],
        "trade_count": int(best["trade_count"]),
        "win_rate": round(best["win_rate"], 1),
        "avg_return_pct": round(best["avg_return_pct"], 2),
        "tier_weight": _TIER_WEIGHT[best["source"]],
        "detail": (
            f"{best['strategy']} @ {best['source']} tier, {best['trade_count']} trades, "
            f"{best['win_rate']:.0f}% win rate, {best['avg_return_pct']:+.2f}%/trade avg"
        ),
    }


def _ma_points(close: float | None, ma_fast: float | None, ma_slow: float | None) -> tuple[float, str]:
    """Price position against the 50/200 MAs, plus the MAs' own relationship.

    Position credit is not a simple ladder from "above everything" down:
    below the fast MA while holding the slow one outranks above-fast-below-slow,
    because the former is a pullback inside an intact uptrend (an entry) and
    the latter is an unproven bounce under long-term resistance.
    """
    if close is None or ma_fast is None or ma_slow is None:
        return _MA_MAX / 2, "not enough history for 50/200 MA yet"

    above_fast, above_slow = close > ma_fast, close > ma_slow
    if above_fast and above_slow:
        pos, pos_label = _MA_POSITION_MAX, "above both the 50 and 200 MA"
    elif above_slow:
        pos, pos_label = _MA_POSITION_MAX * 2 / 3, "pulled back below the 50 MA but holding the 200"
    elif above_fast:
        pos, pos_label = _MA_POSITION_MAX / 2, "reclaimed the 50 MA but still under the 200"
    else:
        pos, pos_label = _MA_POSITION_MAX / 6, "below both the 50 and 200 MA"

    if ma_fast > ma_slow:
        cross, cross_label = _MA_CROSS_MAX, "50 above 200"
    else:
        cross, cross_label = _MA_CROSS_MAX * 3 / 8, "50 below 200"

    return pos + cross, f"{pos_label} ({cross_label})"


def _rsi_points(rsi_value: float | None, regime: str) -> tuple[float, str]:
    """RSI read against the regime.

    In a trend, a pullback into the 30s-40s is the entry and an extreme
    above 80 is exhaustion. In a choppy range the same readings mean the
    opposite thing -- price keeps reverting to the middle, so the low end is
    where a long has edge and the high end is where it gets faded.
    """
    if rsi_value is None:
        return _RSI_MAX / 2, "no RSI reading available"

    if regime == "choppy":
        if rsi_value <= 35:
            pts, label = _RSI_MAX, "oversold in a range -- mean-reversion entry zone"
        elif rsi_value <= 50:
            pts, label = _RSI_MAX * 0.75, "below the middle of a range"
        elif rsi_value <= 65:
            pts, label = _RSI_MAX * 0.45, "above the middle of a range"
        else:
            pts, label = _RSI_MAX * 0.2, "overbought in a range -- poor spot to start a long"
    else:
        if rsi_value < 30:
            pts, label = _RSI_MAX * 0.85, "deeply oversold within a trend -- dip-recovery zone"
        elif rsi_value <= 45:
            pts, label = _RSI_MAX, "pulled back within a trend -- prime entry zone"
        elif rsi_value <= 65:
            pts, label = _RSI_MAX * 0.8, "healthy trend participation"
        elif rsi_value <= 80:
            pts, label = _RSI_MAX * 0.45, "extended"
        else:
            pts, label = _RSI_MAX * 0.2, "exhausted -- blow-off territory"

    return pts, f"RSI {rsi_value:.0f} ({regime}): {label}"


def _macd_points(hist: float | None, hist_prev: float | None) -> tuple[float, str]:
    """MACD histogram sign combined with its direction of travel."""
    if hist is None:
        return _MACD_MAX / 2, "no MACD reading available"
    if hist_prev is None:
        return _MACD_MAX / 2, f"MACD histogram {hist:+.2f}, no prior bar to compare"

    rising = hist > hist_prev
    if hist > 0 and rising:
        pts, label = _MACD_MAX, "positive and expanding -- momentum building"
    elif hist > 0:
        pts, label = _MACD_MAX * 2 / 3, "positive but contracting -- momentum fading"
    elif rising:
        pts, label = _MACD_MAX * 0.55, "negative but contracting -- momentum turning up"
    else:
        pts, label = _MACD_MAX * 0.15, "negative and widening -- momentum deteriorating"

    return pts, f"MACD histogram {hist:+.2f} vs {hist_prev:+.2f}: {label}"


def _volume_points(volume_ratio: float | None, price_change_pct: float | None) -> tuple[float, str]:
    """Volume against its own baseline, interpreted through the direction of
    the price move it accompanied.

    Long-only, so the four quadrants aren't symmetric: an advance wants
    heavy volume behind it (conviction), while a decline wants light volume
    (sellers drying up rather than distributing).
    """
    if volume_ratio is None or price_change_pct is None:
        return _VOLUME_MAX / 2, "no usable volume data"

    heavy = volume_ratio >= _VOLUME_HIGH_RATIO
    light = volume_ratio <= _VOLUME_LOW_RATIO
    advancing = price_change_pct >= 0

    if advancing and heavy:
        pts, label = _VOLUME_MAX, "advance on heavy volume -- conviction"
    elif advancing and light:
        pts, label = _VOLUME_MAX * 0.45, "advance on light volume -- thin, unconvincing"
    elif advancing:
        pts, label = _VOLUME_MAX * 0.7, "advance on average volume"
    elif light:
        pts, label = _VOLUME_MAX * 0.85, "pullback on light volume -- sellers drying up"
    elif heavy:
        pts, label = _VOLUME_MAX * 0.25, "decline on heavy volume -- distribution"
    else:
        pts, label = _VOLUME_MAX * 0.5, "decline on average volume"

    return pts, f"{volume_ratio:.2f}x avg volume, price {price_change_pct:+.1f}% over the same window: {label}"


def _streak_points(streak_bars: int) -> float:
    return min(streak_bars / _STREAK_FULL_CREDIT_BARS, 1.0) * _STREAK_MAX


def _alignment_points(daily_regime: dict, weekly_regime: dict) -> tuple[float, str]:
    d_regime, d_dir = daily_regime["regime"], daily_regime["direction"]
    w_regime, w_dir = weekly_regime["regime"], weekly_regime["direction"]

    if d_dir == w_dir and d_dir not in ("n/a", "flat"):
        return _ALIGNMENT_MAX, f"daily and weekly both {d_dir}"
    if d_regime == "choppy" or w_regime == "choppy":
        return _ALIGNMENT_MAX / 2, f"no conflict, but one side is choppy (daily={d_regime}/{d_dir}, weekly={w_regime}/{w_dir})"
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
    per_side_max = _SECTOR_INDUSTRY_MAX / 2
    if consensus is None:
        return per_side_max / 2, f"no {label} peer data yet"
    if consensus == "mixed":
        return per_side_max / 2, f"{label} peers mixed, no clear majority"
    if self_direction == consensus:
        return per_side_max, f"{label} peers also net {consensus}"
    if self_direction in ("flat", "n/a"):
        return per_side_max / 2, f"{label} peers net {consensus}, no conflict (this ticker is {self_direction})"
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


def _valuation_points(trailing_pe: float | None, forward_pe: float | None) -> tuple[float, str]:
    if trailing_pe is None or forward_pe is None:
        return _VALUATION_MAX / 2, "no usable trailing/forward P/E data (ETF, or negative/missing earnings)"

    gap_pct = (forward_pe - trailing_pe) / trailing_pe * 100
    if gap_pct <= 0:
        points = float(_VALUATION_MAX)
    else:
        points = max(0.0, _VALUATION_MAX * (1 - gap_pct / _VALUATION_STRETCH_THRESHOLD_PCT))

    if gap_pct <= 0:
        detail = f"trailing P/E {trailing_pe:.1f}, forward P/E {forward_pe:.1f} ({-gap_pct:.0f}% lower -- earnings expected to grow)"
    else:
        detail = (
            f"trailing P/E {trailing_pe:.1f}, forward P/E {forward_pe:.1f} "
            f"({gap_pct:.0f}% higher -- earnings expected to shrink, valuation may be stretched)"
        )
    return points, detail


def _technical_components(snapshot: dict, regime: str) -> dict[str, tuple[float, int, str]]:
    """The four technical components as {name: (points, max, detail)}, from
    one technical_snapshot() reading. Shared by the live single-ticker path
    and the DB-driven universe pass so both score identically."""
    ma_pts, ma_detail = _ma_points(snapshot.get("close"), snapshot.get("ma_fast"), snapshot.get("ma_slow"))
    rsi_pts, rsi_detail = _rsi_points(snapshot.get("rsi"), regime)
    macd_pts, macd_detail = _macd_points(snapshot.get("macd_hist"), snapshot.get("macd_hist_prev"))
    vol_pts, vol_detail = _volume_points(snapshot.get("volume_ratio"), snapshot.get("price_change_pct"))
    return {
        "trend_structure": (ma_pts, _MA_MAX, ma_detail),
        "rsi": (rsi_pts, _RSI_MAX, rsi_detail),
        "macd": (macd_pts, _MACD_MAX, macd_detail),
        "volume": (vol_pts, _VOLUME_MAX, vol_detail),
    }


def _grade(total_points: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if total_points >= threshold:
            return grade
    return "D"


def _borderline(total_points: float) -> dict | None:
    """Whether this score sits within _BORDERLINE_MARGIN of a grade cut, and
    which grade is on the other side of it.

    Returns None for scores comfortably inside their band. Otherwise
    `adjacent` is the grade the ticker would hold if it moved across the
    nearest cut, and `direction` says which way it is currently sitting:
    "below" means it fell just short of the better grade (treat it as
    effectively that grade), "above" means it only just holds its own
    (treat it as fragile).
    """
    cuts = [t for t, _ in _GRADE_THRESHOLDS if t > 0]
    nearest = min(cuts, key=lambda c: abs(total_points - c))
    gap = abs(total_points - nearest)
    if gap > _BORDERLINE_MARGIN:
        return None

    above_grade = _grade(nearest)
    below_grade = _grade(nearest - 0.01)
    below = total_points < nearest
    return {
        "cut": float(nearest),
        "gap": round(gap, 1),
        "direction": "below" if below else "above",
        "adjacent": above_grade if below else below_grade,
    }


def compute_setup_score(
    ticker: str,
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    min_trades: int = 15,
    confirm_bars: int = 3,
    news_days: int = 7,
) -> dict:
    """Fetch live data (Tiingo), classify the current regime, read the
    technical snapshot off the latest bar, and combine into an A-D setup
    grade. Also fetches daily+weekly for alignment (reusing the primary
    fetch if `interval` is already one of those), live forward/trailing P/E,
    the historically best-matching strategy and recent news sentiment -- the
    last two as separate context, not part of the score."""
    df = fetch_ohlcv(ticker, period="2y", interval=interval)
    regime_df = confirmed_regime(classify_regime(df), confirm_bars=confirm_bars)
    current = regime_df.iloc[-1]
    streak_bars = regime_streak_bars(regime_df)
    snapshot = technical_snapshot(df)

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

    tech = _technical_components(snapshot, current["regime"])
    streak_pts = _streak_points(streak_bars)
    align_pts, align_detail = _alignment_points(daily_df.iloc[-1], weekly_df.iloc[-1])
    sec_ind_pts, sec_ind_detail = _sector_industry_points(
        daily_df.iloc[-1]["direction"], sector, industry, sector_counts, industry_counts
    )

    try:
        val = fetch_valuation(ticker)
    except Exception:
        val = None
    val_pts, val_detail = _valuation_points(
        val.get("trailing_pe") if val else None, val.get("forward_pe") if val else None
    )

    total = sum(pts for pts, _, _ in tech.values()) + streak_pts + align_pts + sec_ind_pts + val_pts

    news = None
    if news_days > 0:
        try:
            news = ticker_sentiment(ticker, days=news_days)
        except Exception:
            news = None

    components = {name: {"points": round(pts, 1), "max": max_, "detail": detail} for name, (pts, max_, detail) in tech.items()}
    components.update(
        {
            "streak": {"points": round(streak_pts, 1), "max": _STREAK_MAX, "detail": f"{streak_bars} bars in current confirmed regime"},
            "alignment": {"points": round(align_pts, 1), "max": _ALIGNMENT_MAX, "detail": align_detail},
            "sector_industry": {"points": round(sec_ind_pts, 1), "max": _SECTOR_INDUSTRY_MAX, "detail": sec_ind_detail},
            "valuation": {"points": round(val_pts, 1), "max": _VALUATION_MAX, "detail": val_detail},
        }
    )

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "regime": current["regime"],
        "direction": current["direction"],
        "sector": sector,
        "industry": industry,
        "grade": _grade(total),
        "total_points": round(total, 1),
        "borderline": _borderline(round(total, 1)),
        "components": components,
        "strategy": _strategy_info(best),
        "technicals": snapshot,
        "news": news,
    }


def compute_universe_setup_scores(
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    min_trades: int = 15,
) -> pd.DataFrame:
    """Score every ticker the last nightly snapshot run has a stored regime
    for, entirely from the DB -- no live fetching. This is what powers the
    universe-wide setup-grade dashboard: compute_setup_score() above stays
    live-fetch-based (needed for on-demand lookups of tickers outside the
    tracking universe, and for freshness beyond the last snapshot), but
    scoring 400+ tickers one live fetch at a time is slow and redundant --
    the nightly run already classified every tracked ticker's regime, read
    its technical snapshot and refreshed valuation.py's P/E data; this just
    reads that back.

    Every component falls back to half credit when its input is missing for
    a ticker (no technicals row from an older snapshot, missing daily/weekly
    regime, no valuation row) -- consistent with how missing peer data is
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
    valuation_by_ticker = db_module.all_valuation(conn).set_index("ticker")
    technicals = db_module.all_technicals(conn, interval)
    technicals_by_ticker = technicals.set_index("ticker") if not technicals.empty else None

    rows = []
    for _, r in regimes.iterrows():
        ticker, sector, industry = r["ticker"], r["sector"] or "Unknown", r["industry"] or "Unknown"
        best = db_module.best_strategy(
            conn, interval, r["regime"], r["direction"],
            ticker=ticker, industry=industry, sector=sector, min_trades=min_trades,
        )
        if technicals_by_ticker is not None and ticker in technicals_by_ticker.index:
            snapshot = {k: (None if pd.isna(v) else v) for k, v in technicals_by_ticker.loc[ticker].items()}
        else:
            snapshot = {}
        tech = _technical_components(snapshot, r["regime"])
        streak_pts = _streak_points(int(r["streak_bars"]))

        d_row = daily_by_ticker.loc[ticker] if daily_by_ticker is not None and ticker in daily_by_ticker.index else None
        w_row = weekly_by_ticker.loc[ticker] if weekly_by_ticker is not None and ticker in weekly_by_ticker.index else None
        if d_row is not None and w_row is not None:
            align_pts, align_detail = _alignment_points(d_row, w_row)
        else:
            align_pts, align_detail = _ALIGNMENT_MAX / 2, "missing daily or weekly snapshot data, assumed neutral"

        daily_direction = d_row["direction"] if d_row is not None else r["direction"]
        sec_ind_pts, sec_ind_detail = _sector_industry_points(daily_direction, sector, industry, sector_counts, industry_counts)

        if ticker in valuation_by_ticker.index:
            v_row = valuation_by_ticker.loc[ticker]
            val_pts, val_detail = _valuation_points(v_row["trailing_pe"], v_row["forward_pe"])
        else:
            val_pts, val_detail = _VALUATION_MAX / 2, "no valuation data yet for this ticker"

        total = sum(pts for pts, _, _ in tech.values()) + streak_pts + align_pts + sec_ind_pts + val_pts
        row = {
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "regime": r["regime"],
            "direction": r["direction"],
            "grade": _grade(total),
            "total_points": round(total, 1),
            "borderline": _borderline(round(total, 1)),
            "streak_points": round(streak_pts, 1),
            "streak_bars": int(r["streak_bars"]),
            "alignment_points": round(align_pts, 1),
            "alignment_detail": align_detail,
            "sector_industry_points": round(sec_ind_pts, 1),
            "sector_industry_detail": sec_ind_detail,
            "valuation_points": round(val_pts, 1),
            "valuation_detail": val_detail,
            "strategy_info": _strategy_info(best),
        }
        for name, (pts, _, detail) in tech.items():
            row[f"{name}_points"] = round(pts, 1)
            row[f"{name}_detail"] = detail
        rows.append(row)

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
        print(f"  {name:>16}: {c['points']:>5.1f}/{c['max']}  {c['detail']}")
    print(f"  {'strategy to use':>16}: {result['strategy']['detail']}")
    if result["news"] and result["news"]["article_count"] > 0:
        n = result["news"]
        print(f"  news: {n['article_count']} articles, {n['label']} (avg compound {n['avg_compound']:+.3f}), counts: {n['counts']}")
    elif result["news"] is not None:
        print("  news: no recent articles")


if __name__ == "__main__":
    main()
