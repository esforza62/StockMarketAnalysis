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


def vwap_mean_reversion(df: pd.DataFrame, window: int = 20, atr_window: int = 14, band_mult: float = 1.5) -> pd.DataFrame:
    """Fade a stretch away from VWAP rather than follow it -- the "ranging
    market" VWAP play: price extends band_mult ATRs below VWAP, then buy the
    snap back through that lower band. Exit at VWAP itself (fair value),
    not waiting for the upper band -- small, high-frequency wins in a range,
    not a trend-following hold."""
    close = df["Close"]
    v = ind.vwap(df, window)
    lower = v - ind.atr(df, atr_window) * band_mult
    entry = (close > lower) & (close.shift(1) <= lower.shift(1))
    exit_ = (close > v) & (close.shift(1) <= v.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_breakout(df: pd.DataFrame, window: int = 20, atr_window: int = 14, band_mult: float = 1.5, vol_window: int = 20, vol_mult: float = 1.5) -> pd.DataFrame:
    """The opposite read of the same band vwap_mean_reversion fades: a close
    breaking out ABOVE the upper VWAP band, with volume confirmation (above
    vol_mult times its rolling average) so a real directional move is
    distinguished from a stretch that's just going to mean-revert back to
    vwap_mean_reversion's territory. Exit when price gives the breakout back
    by closing under VWAP itself."""
    close = df["Close"]
    v = ind.vwap(df, window)
    upper = v + ind.atr(df, atr_window) * band_mult
    volume_confirmed = df["Volume"] > df["Volume"].rolling(vol_window).mean() * vol_mult
    entry = (close > upper) & (close.shift(1) <= upper.shift(1)) & volume_confirmed
    exit_ = (close < v) & (close.shift(1) >= v.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_pullback(df: pd.DataFrame, window: int = 20, slope_window: int = 5, atr_window: int = 14, touch_mult: float = 0.5) -> pd.DataFrame:
    """Joins an already-established uptrend (VWAP rising, price already
    above it) at a better price than vwap_trend_structure's bare crossover
    entry: waits for a pullback bar whose LOW dips into a tight zone just
    above VWAP (within touch_mult ATRs) while the CLOSE still holds above
    VWAP -- buying the bounce off VWAP-as-support, not a fresh crossunder.
    Exit when price finally closes below VWAP (the support genuinely broke,
    not just got tested)."""
    close, low = df["Close"], df["Low"]
    v = ind.vwap(df, window)
    v_rising = v.diff(slope_window) > 0
    touch_zone = v + ind.atr(df, atr_window) * touch_mult
    pulled_back = low <= touch_zone
    was_trending_above = close.shift(1) > v.shift(1)
    entry = pulled_back & was_trending_above & v_rising & (close > v)
    exit_ = (close < v) & (close.shift(1) >= v.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_ma_cross(df: pd.DataFrame, fast: int = 10, slow: int = 30, window: int = 20) -> pd.DataFrame:
    """Dual confirmation: a fast/slow EMA bullish cross only counts as an
    entry if price is ALSO above VWAP at that moment -- filters out MA
    crosses happening while price is still on the wrong side of fair value.
    Exit on either signal breaking (MA bear cross, or a close back below
    VWAP), whichever comes first."""
    close = df["Close"]
    fast_ma, slow_ma = ind.ema(close, fast), ind.ema(close, slow)
    v = ind.vwap(df, window)
    ma_bull_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    ma_bear_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    vwap_break = (close < v) & (close.shift(1) >= v.shift(1))
    entry = ma_bull_cross & (close > v)
    exit_ = ma_bear_cross | vwap_break
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_opening_range(df: pd.DataFrame, or_bars: int = 2, window: int = 20) -> pd.DataFrame:
    """Session opening-range breakout with VWAP confirmation: each calendar
    day's first or_bars bars set that day's opening range; a later close
    breaking above the range high, while also above VWAP, is the entry.

    Only meaningful on genuinely intraday bars (1h/15m) -- daily/weekly bars
    have no intraday session structure, so each "session" collapses to a
    single bar and this produces zero trades there, the same graceful-thin
    outcome any other strategy gets on a regime bucket it doesn't suit."""
    high, low, close = df["High"], df["Low"], df["Close"]
    v = ind.vwap(df, window)
    session = pd.Series(df.index.date, index=df.index)
    bar_rank = session.groupby(session).cumcount()
    in_opening_range = bar_rank < or_bars
    or_high = high.where(in_opening_range).groupby(session).transform("max")

    after_opening_range = bar_rank >= or_bars
    entry = after_opening_range & (close > or_high) & (close.shift(1) <= or_high.shift(1)) & (close > v)
    exit_ = (close < v) & (close.shift(1) >= v.shift(1))
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def vwap_multi_timeframe(df: pd.DataFrame, window: int = 20, htf_factor: int = 4, slope_window: int = 5) -> pd.DataFrame:
    """Approximates cross-timeframe VWAP confluence using only the single
    OHLCV series already fetched, since this project's backtest harness
    runs one strategy against one interval's data at a time -- a strategy
    function has no access to a second interval's own fetch. The "higher
    timeframe" VWAP here is a rolling VWAP over htf_factor times the base
    window (a coarser, slower view of the same series), not a genuine
    second timeframe's own volume-weighted price -- a true cross-interval
    version would need backtest.py's signal contract extended to accept
    multiple per-interval DataFrames, a larger change than a new strategy
    function, flagged here rather than silently faked.

    Entry: close above BOTH the short and long VWAP, with both rising --
    the immediate crossover and the slower "higher timeframe" read agree.
    Exit: close breaks back below the short VWAP."""
    close = df["Close"]
    v_short = ind.vwap(df, window)
    v_long = ind.vwap(df, window * htf_factor)
    short_rising = v_short.diff(slope_window) > 0
    long_rising = v_long.diff(slope_window) > 0
    entry = (
        (close > v_short) & (close.shift(1) <= v_short.shift(1))
        & (close > v_long) & short_rising & long_rising
    )
    exit_ = (close < v_short) & (close.shift(1) >= v_short.shift(1))
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


def rsi_dip_recovery(
    df: pd.DataFrame,
    rsi_window: int = 14,
    dip_threshold: float = 35.0,
    confirm_threshold: float = 45.0,
    dip_lookback: int = 10,
    exit_rsi: float = 75.0,
) -> pd.DataFrame:
    """Long side only -- see the module-level note on this strategy's short
    side and trailing-stop exit, both dropped as not portable to this
    long-only, fixed-signal engine.

    No trend filter by design: every trade is already tagged with the SMC
    regime active on entry downstream, so which regime this pattern
    actually works in is exactly what the regime-conditioned backtest is
    for -- baking a trend filter into the strategy itself would only hide
    that signal by construction (an earlier HMA-trend-filtered version was
    dropped for exactly this reason).

    Entry: RSI dipping into oversold (< dip_threshold) and then crossing
    back above confirm_threshold -- a wider two-stage band than plain
    rsi_mean_reversion's single 30-threshold cross.
    Exit: RSI reaching overbought (>= exit_rsi).
    """
    r = ind.rsi(df["Close"], rsi_window)

    dipped_recently = (r < dip_threshold).rolling(dip_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    confirm_cross = (r > confirm_threshold) & (r.shift(1) <= confirm_threshold)
    entry = confirm_cross & dipped_recently

    exit_ = (r >= exit_rsi) & (r.shift(1) < exit_rsi)

    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_dip_recovery_trend_filter(
    df: pd.DataFrame,
    rsi_window: int = 14,
    dip_threshold: float = 35.0,
    confirm_threshold: float = 45.0,
    dip_lookback: int = 10,
    exit_rsi: float = 75.0,
    fast_ma: int = 20,
    slow_ma: int = 50,
) -> pd.DataFrame:
    """rsi_dip_recovery with one added gate: only take the entry if the
    fast EMA is still above the slow EMA (still a confirmed uptrend
    structure), everything else identical -- same entry/exit thresholds,
    same long-only design.

    This directly contradicts rsi_dip_recovery's own docstring, which
    argues a trend filter baked into the strategy just hides the signal
    that regime-tagging is supposed to surface, and drops one for that
    reason. The contradiction is deliberate, backed by a real finding
    rather than a stylistic preference: rsi_dip_recovery's daily
    trending/down bucket showed an 81.5% win rate and +14.3% avg
    return/trade -- and also a -98.4% worst-case max drawdown on at least
    one ticker's compounding equity curve (BOIL, NVAX, KOLD all showed
    -90%+ drawdowns), invisible in the headline stats. Every stop-loss
    variant tested (fixed %, ATR-based, time-based, and combinations)
    failed to fix this: capping individual trade losses just means more
    frequent re-entries into the same still-declining ticker, compounding
    down to nearly the same catastrophic drawdown while giving back most
    of the edge. This entry-side filter is what actually worked, verified
    against the full 412-ticker tracking universe (not just a sample):
    worst-case drawdown roughly halved (-98.4% -> -86.3%) while the win
    rate held (81.5% -> 82.2%) and about a third of the edge survived
    (+98.1% -> +29.0% compounded per-ticker return in trending/down) --
    a real, validated risk/edge trade-off, not a free lunch (stacking a
    stop or a max-hold-time cap on top pushes drawdown lower still, but
    at that point gives back nearly all the remaining edge too).
    """
    r = ind.rsi(df["Close"], rsi_window)
    close = df["Close"]
    fast = ind.ema(close, fast_ma)
    slow = ind.ema(close, slow_ma)

    dipped_recently = (r < dip_threshold).rolling(dip_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    confirm_cross = (r > confirm_threshold) & (r.shift(1) <= confirm_threshold)
    entry = confirm_cross & dipped_recently & (fast > slow)

    exit_ = (r >= exit_rsi) & (r.shift(1) < exit_rsi)

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
    variant of rsi_dip_recovery, with an HMA(25) trend filter still gating
    entry (unlike rsi_dip_recovery, which dropped its HMA filter as
    redundant with the downstream regime tagging -- worth reconsidering
    here too).

    Entry: slow RSI(21) > midline confirms the broader bullish bias; fast
    RSI(5) crossing down through midline is the trigger -- buying the start
    of a dip within a regime the slow RSI says is still bullish, rather
    than buying the dip's resolution (which is what rsi_dip_recovery
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
    "vwap_mean_reversion": vwap_mean_reversion,
    "vwap_breakout": vwap_breakout,
    "vwap_pullback": vwap_pullback,
    "vwap_ma_cross": vwap_ma_cross,
    "vwap_opening_range": vwap_opening_range,
    "vwap_multi_timeframe": vwap_multi_timeframe,
    "rsi_macd_reversal": rsi_macd_reversal,
    "rsi_macd_trend": rsi_macd_trend_continuation,
    "rsi_macd_filter": rsi_macd_filter,
    "rsi_dip_recovery": rsi_dip_recovery,
    "rsi_dip_trend_filter": rsi_dip_recovery_trend_filter,
    "rsi_dual_hma": rsi_dual_hma_trend,
}
