"""Signal generators for candidate trading strategies.

Each function takes OHLCV and returns a DataFrame with boolean "entry" and
"exit" columns, aligned to the input index, for a single-position long-only
strategy. Used by backtest.py to build trade logs, and by regime_backtest.py
to tag those trades with the SMC regime active on entry.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind


def rsi_mean_reversion(df: pd.DataFrame, window: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.DataFrame:
    r = ind.rsi(df["Close"], window)
    entry = (r > oversold) & (r.shift(1) <= oversold)
    exit_ = (r > overbought) & (r.shift(1) <= overbought)
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def bollinger_mean_reversion(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    bands = ind.bollinger_bands(df["Close"], window, num_std)
    close = df["Close"]
    entry = (close > bands["lower"]) & (close.shift(1) <= bands["lower"].shift(1))
    exit_ = (close > bands["mid"]) & (close.shift(1) <= bands["mid"].shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def macd_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    m = ind.macd(df["Close"], fast, slow, signal)
    entry = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))
    exit_ = (m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def ema_trend_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    close = df["Close"]
    fast_ema = ind.ema(close, fast)
    slow_ema = ind.ema(close, slow)
    entry = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    exit_ = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def donchian_breakout(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    channel = ind.donchian_channel(df, window)
    close = df["Close"]
    prior_upper = channel["upper"].shift(1)
    prior_lower = channel["lower"].shift(1)
    entry = close > prior_upper
    exit_ = close < prior_lower
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def supertrend_following(df: pd.DataFrame, atr_window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    st = ind.supertrend(df, atr_window, multiplier)
    direction = st["direction"]
    entry = (direction > 0) & (direction.shift(1) <= 0)
    exit_ = (direction < 0) & (direction.shift(1) >= 0)
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_trend_structure(df: pd.DataFrame, window: int = 20, slope_window: int = 5) -> pd.DataFrame:
    """Price crossing above a rising rolling VWAP -- trend *structure*, not a
    bare crossover, since the slope filter requires VWAP itself to be
    trending up (down) for a long entry (exit)."""
    close = df["Close"]
    v = ind.vwap(df, window)
    v_rising = v.diff(slope_window) > 0
    entry = (close > v) & (close.shift(1) <= v.shift(1)) & v_rising
    exit_ = (close < v) & (close.shift(1) >= v.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_macd_reversal(df: pd.DataFrame, window: int = 14, oversold: float = 30.0, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Oversold RSI + bullish MACD cross -- momentum confirms a reversal off
    an extreme low, rather than mean-reverting on RSI alone."""
    r = ind.rsi(df["Close"], window)
    m = ind.macd(df["Close"], fast, slow, signal)
    macd_bull_cross = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))
    macd_bear_cross = (m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))
    entry = macd_bull_cross & (r < oversold)
    exit_ = macd_bear_cross
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_macd_trend_continuation(df: pd.DataFrame, window: int = 14, band_low: float = 40.0, band_high: float = 70.0, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """RSI holding in a healthy trend band (not overbought/oversold) confirms
    the trend isn't exhausted, while a re-accelerating MACD histogram --
    troughing and turning back up -- is the actual entry trigger."""
    r = ind.rsi(df["Close"], window)
    hist = ind.macd(df["Close"], fast, slow, signal)["histogram"]
    hist_rising = hist.diff() > 0
    hist_reaccelerating = hist_rising & ~hist_rising.shift(1).fillna(False)
    hist_decelerating = ~hist_rising & hist_rising.shift(1).fillna(False)
    in_band = r.between(band_low, band_high)
    entry = hist_reaccelerating & in_band
    exit_ = hist_decelerating
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_macd_filter(df: pd.DataFrame, window: int = 14, midline: float = 50.0, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """RSI above/below its midline filters MACD crosses to only the ones
    that agree with the prevailing momentum bias, instead of trading every
    MACD cross regardless of context."""
    r = ind.rsi(df["Close"], window)
    m = ind.macd(df["Close"], fast, slow, signal)
    macd_bull_cross = (m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))
    macd_bear_cross = (m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))
    entry = macd_bull_cross & (r > midline)
    exit_ = macd_bear_cross
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_hma_trend(
    df: pd.DataFrame,
    hma_window: int = 25,
    hma_slope_lookback: int = 3,
    rsi_window: int = 14,
    dip_threshold: float = 35.0,
    confirm_threshold: float = 45.0,
    dip_lookback: int = 10,
    exit_rsi: float = 75.0,
) -> pd.DataFrame:
    """Long side only -- see the module-level note on rsi_hma_trend's short
    side and trailing-stop exit, both dropped as not portable to this
    long-only, fixed-signal engine.

    Entry: price above a rising HMA (trend filter) confirms the setup;
    RSI dipping into oversold (< dip_threshold) and then crossing back
    above confirm_threshold is the actual trigger -- a pullback-in-uptrend
    entry, not a straight oversold bounce.
    Exit: RSI reaching overbought (>= exit_rsi), or price closing back
    below the HMA (trend invalidation) -- whichever comes first.
    """
    close = df["Close"]
    hma = ind.hull_moving_average(close, hma_window)
    hma_rising = hma > hma.shift(hma_slope_lookback)
    r = ind.rsi(close, rsi_window)

    dipped_recently = (r < dip_threshold).rolling(dip_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    confirm_cross = (r > confirm_threshold) & (r.shift(1) <= confirm_threshold)
    entry = (close > hma) & hma_rising & confirm_cross & dipped_recently

    exit_rsi_hit = (r >= exit_rsi) & (r.shift(1) < exit_rsi)
    exit_below_hma = (close < hma) & (close.shift(1) >= hma.shift(1))
    exit_ = exit_rsi_hit | exit_below_hma

    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_dual_hma_trend(
    df: pd.DataFrame,
    hma_window: int = 25,
    hma_slope_lookback: int = 3,
    slow_rsi_window: int = 21,
    fast_rsi_window: int = 5,
    midline: float = 50.0,
) -> pd.DataFrame:
    """Long side only -- dual RSI (21-period "regime" + 5-period "trigger")
    variant of rsi_hma_trend, same HMA(25) trend filter.

    Entry: slow RSI(21) > midline confirms the broader bullish bias; fast
    RSI(5) crossing down through midline is the trigger -- buying the start
    of a dip within a regime the slow RSI says is still bullish, rather
    than buying the dip's resolution (which is what plain rsi_hma_trend
    does). Price above a rising HMA still gates both.
    Exit: slow RSI(21) crossing back below midline (bullish thesis broken)
    or price closing back below the HMA -- deliberately NOT using the fast
    RSI for exit, since a 5-period RSI is too twitchy and would trigger on
    the very pullback noise this entry is trying to buy.
    """
    close = df["Close"]
    hma = ind.hull_moving_average(close, hma_window)
    hma_rising = hma > hma.shift(hma_slope_lookback)
    slow_r = ind.rsi(close, slow_rsi_window)
    fast_r = ind.rsi(close, fast_rsi_window)

    fast_dip_cross = (fast_r < midline) & (fast_r.shift(1) >= midline)
    entry = fast_dip_cross & (slow_r > midline) & (close > hma) & hma_rising

    slow_exit_cross = (slow_r < midline) & (slow_r.shift(1) >= midline)
    exit_below_hma = (close < hma) & (close.shift(1) >= hma.shift(1))
    exit_ = slow_exit_cross | exit_below_hma

    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


STRATEGIES = {
    "rsi": rsi_mean_reversion,
    "bollinger": bollinger_mean_reversion,
    "macd": macd_crossover,
    "ema_cross": ema_trend_cross,
    "donchian": donchian_breakout,
    "supertrend": supertrend_following,
    "vwap_trend": vwap_trend_structure,
    "rsi_macd_reversal": rsi_macd_reversal,
    "rsi_macd_trend": rsi_macd_trend_continuation,
    "rsi_macd_filter": rsi_macd_filter,
    "rsi_hma": rsi_hma_trend,
    "rsi_dual_hma": rsi_dual_hma_trend,
}
