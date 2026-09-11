"""Signal generators for candidate trading strategies.

Each function takes OHLCV and returns a DataFrame with boolean "entry" and
"exit" columns, aligned to the input index, for a single-position long-only
strategy. Used by backtest.py to build trade logs, and by regime_backtest.py
to tag those trades with the SMC regime active on entry.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind
from . import patterns as pat


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


def sweep_outside_reversal(
    df: pd.DataFrame,
    wick_body_mult: float = 2.0,
    wick_range_frac: float = 0.5,
    require_sweep: bool = True,
    atr_window: int = 14,
    min_range_atr: float = 0.5,
    slot_normalized_atr: bool = True,
    require_close_beyond: bool = True,
    middle_colour: str = "same",
    confirm: str = "first_extreme",
) -> pd.DataFrame:
    """Price-structure reversal on either of two patterns firing -- the
    first strategy here driven by bar structure rather than an indicator.

    Entry: EITHER a bullish three-bar sweep-and-reclaim (two red candles,
    the second sweeping the first's low with a long lower wick, then a
    close back above the first's high) OR a bullish outside bar (range
    engulfs the prior bar, close beyond its high). Both are the same read
    -- a level was taken, rejected, and reclaimed on the close -- over
    three bars and two respectively, so OR-ing them widens coverage of one
    idea rather than mixing two.
    Exit: either pattern's bearish mirror.

    Long only, like every strategy here: the bearish patterns are the exit,
    not a short entry. The short side is a genuinely useful signal this
    throws away -- the same reason rsi_dip_recovery notes for dropping its
    own -- but taking it needs backtest.py's single-position long-only loop
    extended to short entries, stops above entry, and same-bar reversal (a
    stop-and-reverse strategy goes flat and loses the signal otherwise),
    which is an engine change, not a strategy one.

    No trend filter, deliberately -- see patterns.py's module docstring:
    regime tagging downstream is what answers "which regime does this work
    in," and a filter here would answer it by construction. Higher
    timeframes are treated the same way: smc_regime.timeframes builds the
    15m/1h/4h/1d ladder and smc_regime.mtf_cli reports outcomes split by
    how many rungs agreed, rather than this function gating on agreement
    that has not been shown to help yet.

    Signal thresholds are ATR-relative against a per-time-of-day baseline
    (patterns.slot_atr), which matters on any intraday interval and is a
    no-op on daily bars.

    MEASURED, AND IT DOES NOT BEAT RANDOM. Over 76 tickers of daily bars
    since 2019 the bullish pattern returns 1.28% per month held against
    1.78% for random entries drawn with the same holding profile, and 3.56%
    for buy-and-hold. The headline win rate (66%) and mean trade (+25%) are
    holding-period effects, not selection: all six middle_colour/confirm
    combinations land between 1.63 and 1.98% per month held once hold time
    is divided out, despite trade counts differing eightfold. Whether any
    regime bucket is different is what the nightly regime-conditioned run
    exists to answer -- this pools every regime together, which is exactly
    the averaging that harness was built to avoid.
    """
    signals = pat.reversal_signals(
        df,
        wick_body_mult=wick_body_mult,
        wick_range_frac=wick_range_frac,
        require_sweep=require_sweep,
        atr_window=atr_window,
        min_range_atr=min_range_atr,
        slot_normalized_atr=slot_normalized_atr,
        require_close_beyond=require_close_beyond,
        middle_colour=middle_colour,
        confirm=confirm,
    )
    return pd.DataFrame(
        {"entry": signals["bullish"].fillna(False), "exit": signals["bearish"].fillna(False)}
    )


def reversal_mean_reversion(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: float = 40.0,
    stretch_lookback: int = 3,
    mean_window: int = 20,
    exit_on: str = "mean",
    exit_rsi: float = 70.0,
    **pattern_kwargs,
) -> pd.DataFrame:
    """A bullish reversal bar, but only where price is already stretched.

    The two ingredients were measured separately here first and neither
    result was neutral. sweep_outside_reversal's patterns do NOT beat random
    entries with the same holding profile -- a reversal shape says the last
    three bars resolved upward and nothing about where that happened. Mean
    reversion is the opposite: excess_cli found rsi in choppy at +5.46%
    excess per trade (t = 2.44), and rsi in every other regime at nothing
    much. So this asks a narrow question -- does the SHAPE add anything on
    top of the LOCATION? Entry needs both: RSI at or below `oversold` (the
    stretch) and a bullish sweep-reclaim or outside bar (the turn).

    It differs from rsi_dip_recovery in what confirms the turn, not in what
    it trades. That strategy waits for the INDICATOR to recover back through
    a threshold; this waits for a BAR to reject the low while the indicator
    is still stretched.

    MEASURING THE STRETCH BEFORE THE SIGNAL BAR IS NOT A FUDGE, IT IS THE
    ONLY WAY THE QUESTION IS ASKABLE. A bullish sweep-reclaim closes above
    the first candle's high -- it is a rally by construction, and that rally
    lifts RSI out of oversold on the very bar the pattern completes. Across
    AAPL/NVDA/SPY/KHC since 2019 the confirmation bar's own RSI has a MEDIAN
    near 59 and clears 35 on at most one signal per ticker, so the same-bar
    conjunction is not a strict strategy but an empty one (literally zero
    trades on AAPL). Taking the minimum RSI over the preceding
    `stretch_lookback` bars asks what was intended: was price stretched
    going INTO this reversal. The gate still keeps only about one bullish
    pattern in six.

    THE RESULT: THE BEST ENTRY SIGNAL MEASURED HERE, AND THE WORST STRATEGY.
    Both are true. Over 409 tickers of daily bars since 2019 (Yahoo
    adjusted), against the excess_cli baseline -- the mean return of holding
    the same number of bars from any bar in the same regime:

      pooled          3852 trades  +1.31% vs +1.06% baseline  excess +0.26%  t = 1.40
      choppy           844 trades                             excess +1.02%  t = 3.46
      trending/down   2659 trades                             excess -0.04%  t = -0.51

    The choppy cell is the cleanest this project has produced: 362 tickers,
    65% of them positive, p ~= 0.0005, which survives a Bonferroni
    correction over the ~30 cells examined (threshold 0.0017). The
    trending/down cell is the more USEFUL finding -- it is the only
    mean-reversion variant here that does not bleed in downtrends, against
    rsi_dip_recovery at -0.95% (t = -2.17), rsi at -0.65% (t = -2.10) and
    sweep_outside at -0.80% (t = -2.11). Requiring a bar that reclaims the
    swept low is what separates "oversold and turning" from "oversold and
    still going", which RSI alone cannot do.

    AND IT MAKES NO MONEY. Market exposure is 5.7% -- 9.5 trades per ticker
    averaging 11 bars -- so the strong cell fires 0.30 times per ticker per
    year at +1.02%, or about 0.31% of annual alpha. Median CAGR is 1.4%
    against 14.4% for holding the stock and 17.2% for SPY, and ZERO of 409
    tickers beat the index. Nor is there a knob: exit_on="rsi" lifts median
    CAGR to 5.6% but drops the choppy t to 1.60, a looser 45/5 gate nearly
    doubles trades for excess +0.23% (t = 1.21), and loosening AND holding
    longer reaches 7.3% CAGR at excess -0.02%. Every route to more money
    costs the edge, which is itself evidence the edge is specific and thin.
    Note too that plain rsi in chop scores +5.46% per trade against this
    +1.02%: the pattern gate buys consistency (higher t, more tickers
    positive) and pays for it in magnitude.

    Read that CAGR comparison carefully before dismissing the signal: it
    charges 0% for time in cash, so it structurally punishes anything with
    low exposure. The honest reading is "this is not a way to be invested",
    not "the entries are bad". The shape of the result -- excellent timing,
    negligible exposure -- is a timing overlay, not a strategy: a trigger
    for a name already worth owning rather than something that picks what to
    own. No costs or slippage are modelled anywhere above, and an 11-bar
    average hold is far more sensitive to that than rsi_dip_recovery's 165.

    exit_on="mean" closes when price closes back above its `mean_window`
    EMA -- the reversion trade's own thesis, completed -- rather than
    rsi_dip_recovery's overbought exit, which turns a reversion trade into a
    trend trade. "rsi" reproduces that overbought exit for comparison and
    "pattern" exits on the bearish mirror, as sweep_outside_reversal does.

    Long only, like every strategy here. Pattern keyword arguments pass
    through to patterns.reversal_signals untouched.
    """
    if exit_on not in ("mean", "rsi", "pattern"):
        raise ValueError(f"exit_on must be mean/rsi/pattern, got {exit_on!r}")

    r = ind.rsi(df["Close"], rsi_window)
    signals = pat.reversal_signals(df, **pattern_kwargs)
    mean = ind.ema(df["Close"], mean_window)

    # The stretch is measured BEFORE the confirmation bar, never on it.
    # A bullish sweep-reclaim closes above the first candle's high -- it is
    # a rally by definition, and that rally lifts RSI out of oversold on
    # the very bar the pattern completes. Measured across AAPL/NVDA/SPY/KHC
    # since 2019, the confirmation bar's own RSI has a MEDIAN near 59 and
    # clears 35 on at most one signal per ticker, so the naive same-bar
    # conjunction is not a strict strategy -- it is an empty one. Taking
    # the minimum RSI over the preceding `stretch_lookback` bars asks the
    # question that was actually intended: was price stretched going INTO
    # this reversal.
    stretched = r.rolling(stretch_lookback).min().shift(1) <= oversold
    entry = signals["bullish"] & stretched

    if exit_on == "mean":
        # Crossing back up through the mean, not merely sitting above it:
        # without the shift the exit is true on every bar of a position
        # entered above its own mean, which cannot happen here (an oversold
        # RSI and a close above the EMA rarely coincide) but would silently
        # become an immediate-exit bug if `oversold` were ever loosened.
        exit_ = (df["Close"] >= mean) & (df["Close"].shift(1) < mean.shift(1))
    elif exit_on == "rsi":
        exit_ = (r >= exit_rsi) & (r.shift(1) < exit_rsi)
    else:
        exit_ = signals["bearish"]

    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def rsi_dip_regime_gated(
    df: pd.DataFrame,
    rsi_window: int = 14,
    dip_threshold: float = 35.0,
    confirm_threshold: float = 45.0,
    dip_lookback: int = 10,
    exit_rsi: float = 75.0,
    gated_regimes: tuple[str, ...] = ("choppy",),
    pattern_lookback: int = 1,
    confirm_bars: int = 3,
    **pattern_kwargs,
) -> pd.DataFrame:
    """rsi_dip_recovery, but requiring a reversal bar only in chop.

    MEASURED, AND IT DOES NOT IMPROVE rsi_dip_recovery. Kept because the
    way it fails is the most legible demonstration in this repo of the trap
    that also caught rsi_dip_trend_filter: a gate that raises per-trade
    quality while destroying total return, which looks like an edge from
    every angle except the one that counts.

    The motivation was real. Over 409 tickers of daily bars since 2019,
    requiring a bullish reversal bar on rsi_dip_recovery's entry moves the
    choppy cell from +2.74% excess (t = 0.67) to +7.51% (t = 2.80) while
    taking trending/up from +5.62% (t = 3.03) to -2.20% (t = -0.13). One
    rule, opposite signs by regime, and applied unconditionally a clear net
    loss: -3.02 points of excess paired by ticker (t = -2.24) and a median
    6-point CAGR cut, worse on 84% of names. Gating it to chop only is the
    obvious repair.

    IT DOES NOT REPAIR IT. Pooled, the gated variant is slightly worse than
    the strategy it gates: excess -0.14% against -0.03%, median CAGR 9.3%
    against 9.8%, 18% of tickers beating SPY against 19%. Paired by ticker
    -- the only fair test, since both run the same names -- excess moves
    -0.15 points (t = -0.45, a wash) and CAGR is significantly worse:
    t = -3.87, better on 18% of tickers, unchanged on 47% (the names with
    no chop entries at all).

    WHY THE CHOP CELL LOOKS BETTER THE WHOLE TIME. Tightening the gate
    raises per-trade excess monotonically and shrinks the total:

        no gate                310 chop trades   +2.74%    848 total points
        pattern within 5        52 chop trades  +12.00%    624 total points
        pattern within 3        42 chop trades  +12.17%    511 total points

    A +12.00% mean at t = 4.49 is the best-looking cell this project has
    produced and it is worth LESS than the +2.74% it replaced. Per-trade
    excess rising as sample falls is the signature of selection, not edge;
    the quantity that has to go up is the sum.

    One engine artifact to know about when reading per-regime counts: this
    backtest holds one position at a time, so blocking a chop entry frees
    the slot for a later one elsewhere. Gating chop RAISES trending/down
    from 2090 to 2231 trades and trending/up from 219 to 236. Per-regime
    trade counts are therefore not independent across regimes, and a gate
    on one regime silently rewrites the others.

    THIS ALSO BREAKS THIS MODULE'S OWN RULE, deliberately. Every other
    strategy here refuses to look at the regime, because each trade is
    tagged with the regime active on entry downstream and gating internally
    answers "which regime does this work in" by construction --
    rsi_dip_recovery says so in its docstring and an HMA-filtered version
    was dropped over it. The exception was made because the motivating
    measurement was regime-split rather than pooled, and there is no single
    answer to "should this gate be on". The price is that this strategy's
    own per-regime results are no longer an independent measurement of
    anything: the chop cell is a different strategy from the trending cells
    by construction, and it was selected for. Judge it against
    rsi_dip_recovery paired by ticker on pooled numbers, which is what the
    verdict above does.

    The regime is the confirmed one (regime.confirmed_regime, hysteresis
    over `confirm_bars`), so it is causal -- bar t's label uses bars up to t
    and nothing after. `pattern_lookback` of 1 requires the pattern on the
    entry bar itself; higher values accept one within that many bars.
    """
    from .regime import RegimeThresholds, classify_regime, confirmed_regime

    r = ind.rsi(df["Close"], rsi_window)
    dipped_recently = (
        (r < dip_threshold).rolling(dip_lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
    )
    entry = (r > confirm_threshold) & (r.shift(1) <= confirm_threshold) & dipped_recently

    regime = confirmed_regime(
        classify_regime(df, RegimeThresholds()), confirm_bars=confirm_bars
    )["regime"].reindex(df.index)
    gated = regime.isin(gated_regimes).fillna(False)

    bullish = pat.reversal_signals(df, **pattern_kwargs)["bullish"]
    if pattern_lookback > 1:
        bullish = bullish.rolling(pattern_lookback, min_periods=1).max().astype(bool)

    # Outside the gated regimes the entry is rsi_dip_recovery's, untouched.
    entry = entry & (~gated | bullish.fillna(False))

    exit_ = (r >= exit_rsi) & (r.shift(1) < exit_rsi)
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def band_reversal_oversold(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    rsi_window: int = 14,
    oversold: float = 35.0,
    rsi_lookback: int = 3,
    exit_at: str = "opposite",
    **pattern_kwargs,
) -> pd.DataFrame:
    """A reversal bar that pierced the lower 2-sigma band while RSI was oversold.

    Entry needs all three: a bullish sweep-reclaim or outside bar (the turn),
    the bar's LOW below the lower Bollinger band (the location), and RSI
    already at an extreme (the momentum state). Exit at the opposite band.

    MEASURE THE LOCATION ON THE LOW, THE RSI OFF THE SIGNAL BAR. The pattern's
    confirming bar closes above the prior bar's high -- it is a rally by
    construction, so it drags price back inside the band and lifts RSI out of
    oversold on the very bar the pattern completes. A close-based band test
    or a same-bar RSI test rejects almost everything it is meant to select;
    reversal_mean_reversion documents the same failure costing it every trade
    on AAPL. Hence `Low` against the band, and RSI's minimum over the
    preceding `rsi_lookback` bars.

    THE RESULT, 409 tickers of 4h bars (Yahoo hourly aggregated through
    timeframes.session_resample), 2024-09-16 to 2026-09-10, ~989 bars each,
    798 trades. Scored against SPY over each trade's OWN holding window,
    which is the benchmark that fits a scanner -- the setup is not married to
    a ticker, so the alternative use of the capital is the index, not
    buy-and-hold of the same name:

        setup    3.35% per trade vs SPY's 2.33%   alpha +1.01%  t = 2.39
        random  -0.24% alpha, matched holding profile, same tickers

        beat SPY on 65% of trades; matched random entries beat it on 46%

    The RAW win rate is 72%, and it means nothing on its own: random entries
    in the same names over the same durations won 52%, because the window was
    a rising market. The 19-point gap in BEAT-SPY rate is the edge.

    Against the excess_cli baseline instead (mean forward return of holding
    the same bars from any bar in the same regime) it is +1.67% per trade,
    t = 3.40 clustered by ticker, and total excess rises with the filter --
    332 to 1333 points -- which is what distinguishes a real filter from
    the sample-shrinking kind (see rsi_dip_regime_gated for the other kind).

    OUT OF SAMPLE, split at 2025-09-15 with no retuning: H1 +1.58% excess
    (t = 2.01, 445 trades), H2 +1.79% (t = 2.30, 353). Paired against the
    unfiltered band entry within each half: +1.63 points (t = 3.86) and
    +1.21 (t = 2.65). Both halves, same sign, similar size.

    SLIPPAGE HALVES IT BY 25 BPS A SIDE. Alpha runs +1.01% frictionless,
    +0.91% (t = 2.14) at 5 bps, +0.81% (t = 1.88) at 10 bps, +0.50%
    (t = 1.12) at 25 bps. Tradeable at large-cap liquidity; probably not in
    thin names, where a ~40-bar 4h hold still pays the spread twice.

    WHAT IS NOT ESTABLISHED. Run as a portfolio across the universe it
    supports only about 4-8 concurrent positions -- capital is 83% deployed
    at 5 slots and 54% at 40 -- and it clears SPY in both halves ONLY at the
    5-slot configuration, on 53 and 62 trades, with wildly unstable margins
    (+3.8 then +19.2 points). Two flaws make even that optimistic: the
    drawdowns behind it are marked at entry events with open positions held
    at cost, so they are floors rather than maxima; and with 683 of 798
    signals skipped for want of a slot, WHICH ones get taken is decided by
    arrival order, not by quality. Treat the per-trade alpha as the finding
    and the portfolio numbers as unvalidated.

    The edge is also carried by magnitude, not breadth -- it beats the
    unfiltered entry on only 45% and 42% of tickers per half while winning
    on average, so it will look wrong on most individual names.

    INTERVAL-SPECIFIC, like every bar-shape signal here. On daily bars the
    same rule is worth +0.02% excess. patterns.py explains why a result on
    one interval says nothing about another. This is a 4h finding.

    exit_at="opposite" closes at the upper band; "basis" closes at the
    middle band, which is the intuitive mean-reversion target and measurably
    too early -- same entries, +0.19% excess against -0.02% on 4h, and
    +0.79% against +0.02% on daily.
    """
    if exit_at not in ("opposite", "basis"):
        raise ValueError(f"exit_at must be opposite/basis, got {exit_at!r}")

    bb = ind.bollinger_bands(df["Close"], window, num_std)
    r = ind.rsi(df["Close"], rsi_window)
    bullish = pat.reversal_signals(df, **pattern_kwargs)["bullish"]

    entry = bullish & (df["Low"] < bb["lower"]) & (r.rolling(rsi_lookback).min() <= oversold)
    if exit_at == "opposite":
        exit_ = df["Close"] >= bb["upper"]
    else:
        exit_ = (df["Close"] >= bb["mid"]) & (df["Close"].shift(1) < bb["mid"].shift(1))
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
    "reversal_mean_reversion": reversal_mean_reversion,
    "rsi_dip_regime_gated": rsi_dip_regime_gated,
    "band_reversal_oversold": band_reversal_oversold,
    "sweep_outside": sweep_outside_reversal,
}
