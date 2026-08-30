"""Per-ticker technical snapshot taken on the LAST bar of a price series.

This is the raw material for the setup score's technical components (RSI,
MACD, volume conviction, price vs. the 50/200 MAs). It is deliberately a
plain "what do the indicators read right now" dump with no scoring logic --
setup_score.py owns the interpretation, this owns the measurement.

Captured as a byproduct of the nightly backtest's existing per-ticker fetch
(see regime_backtest.collect_trades) and persisted to the `technicals`
table, so the universe-wide scorer can read it back without re-fetching
price data for 400+ tickers.

Every field is optional: short series (a 200-bar MA needs 200 bars) and
missing volume simply yield None, and the scorer treats None as neutral
rather than as evidence against a setup.
"""
from __future__ import annotations

import pandas as pd

from .indicators import macd, rsi

# Recent-vs-baseline volume comparison. 5 bars is "the last week" on daily
# bars -- long enough not to swing on one heavy print, short enough to still
# describe current participation; 50 is the conventional baseline window.
_VOLUME_RECENT_BARS = 5
_VOLUME_BASELINE_BARS = 50
_MA_FAST = 50
_MA_SLOW = 200


def _last_float(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def technical_snapshot(df: pd.DataFrame) -> dict:
    """Read RSI, MACD, volume participation and 50/200-MA position off the
    final bar of `df`.

    `macd_hist_prev` is returned alongside `macd_hist` so a consumer can tell
    an expanding histogram (momentum building) from a contracting one
    (momentum fading) -- the sign alone doesn't distinguish those, and the
    difference matters more than the level for an entry decision.

    `volume_ratio` is recent volume over its own longer baseline, and
    `price_change_pct` is the price move across that same recent window, so
    the two can be read together as conviction: the same 1.5x volume means
    something very different behind an advance than behind a decline.
    """
    close = df["Close"]
    snapshot: dict = {
        "close": _last_float(close),
        "rsi": _last_float(rsi(close)),
        "ma_fast": _last_float(close.rolling(_MA_FAST).mean()),
        "ma_slow": _last_float(close.rolling(_MA_SLOW).mean()),
        "macd_hist": None,
        "macd_hist_prev": None,
        "volume_ratio": None,
        "price_change_pct": None,
    }

    hist = macd(close)["histogram"]
    snapshot["macd_hist"] = _last_float(hist)
    snapshot["macd_hist_prev"] = _last_float(hist.iloc[:-1]) if len(hist) > 1 else None

    if "Volume" in df.columns and len(close) > _VOLUME_RECENT_BARS:
        volume = df["Volume"]
        recent = volume.tail(_VOLUME_RECENT_BARS).mean()
        baseline = volume.tail(_VOLUME_BASELINE_BARS).mean()
        if pd.notna(recent) and pd.notna(baseline) and baseline > 0:
            snapshot["volume_ratio"] = float(recent / baseline)

        past = close.iloc[-(_VOLUME_RECENT_BARS + 1)]
        if pd.notna(past) and past != 0:
            snapshot["price_change_pct"] = float((close.iloc[-1] - past) / past * 100)

    return snapshot
