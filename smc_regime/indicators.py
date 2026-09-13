"""Technical indicators used for price-movement regime classification."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, adjust=False).mean()


def atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR as a percentage of closing price."""
    return atr(df, window) / df["Close"] * 100


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index -- classic trend-strength measure (0-100)."""
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr_smooth = true_range(df).ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_smooth

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    result = dx.ewm(alpha=1 / window, adjust=False).mean()

    # ADX is a doubly-smoothed value (TR->ATR, then DX->ADX), so it needs
    # roughly 2*window bars before the smoothing has actually accumulated
    # enough data to mean anything. Unlike efficiency_ratio/rolling_slope
    # (which use .rolling(window) and correctly return NaN until a full
    # window exists), .ewm() happily produces a value from the very first
    # bar -- e.g. a single one-directional move right at the start of a
    # fetch window can make dx compute to exactly 100, which then reads as
    # a maximally strong trend before the indicator has any real basis for
    # that. Mask it out explicitly so early-window noise can't masquerade
    # as a trend signal.
    result.iloc[: min(2 * window - 1, len(result))] = np.nan
    return result


def efficiency_ratio(close: pd.Series, window: int = 14) -> pd.Series:
    """Kaufman Efficiency Ratio: net move / total path length, in [0, 1].

    Near 1 = price moved efficiently in one direction (trending).
    Near 0 = price churned back and forth without net progress (choppy).
    """
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return (net_change / path_length.replace(0, np.nan)).clip(0, 1)


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average -- the most recent bar in the
    window gets the highest weight, the oldest the lowest."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_moving_average(close: pd.Series, window: int = 20) -> pd.Series:
    """Hull Moving Average (Alan Hull): WMA(2*WMA(n/2) - WMA(n), sqrt(n)).

    Reduces the lag of a plain WMA/EMA by overweighting the most recent
    half-window move, then re-smoothing over a shorter sqrt(n) window --
    tracks price more tightly through turns while still filtering noise.
    """
    half_window = max(1, window // 2)
    sqrt_window = max(1, round(window ** 0.5))
    raw_hma = 2 * wma(close, half_window) - wma(close, window)
    return wma(raw_hma, sqrt_window)


def ema_extension_atr_units(df: pd.DataFrame, ema_window: int = 20, atr_window: int = 14) -> pd.Series:
    """Signed distance of close from its EMA, expressed in ATR units."""
    close = df["Close"]
    dist = close - ema(close, ema_window)
    return dist / atr(df, atr_window).replace(0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line})


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std})


def donchian_channel(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"upper": df["High"].rolling(window).max(), "lower": df["Low"].rolling(window).min()})


def supertrend(df: pd.DataFrame, atr_window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, atr_window)
    close = df["Close"]
    n = len(df)

    final_upper = (hl2 + multiplier * atr_val).to_numpy().copy()
    final_lower = (hl2 - multiplier * atr_val).to_numpy().copy()
    close_arr = close.to_numpy()
    direction = np.ones(n)

    for i in range(1, n):
        if close_arr[i - 1] <= final_upper[i - 1]:
            final_upper[i] = min(final_upper[i], final_upper[i - 1])
        if close_arr[i - 1] >= final_lower[i - 1]:
            final_lower[i] = max(final_lower[i], final_lower[i - 1])

        if direction[i - 1] == 1:
            direction[i] = -1 if close_arr[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close_arr[i] > final_upper[i] else -1

    line = np.where(direction == 1, final_lower, final_upper)
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=df.index)


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling volume-weighted average price over a fixed window.

    Not a session-anchored VWAP (no intraday session boundaries here, and no
    fixed anchor-event rule) -- a rolling approximation, which is the
    standard substitute when working from daily/hourly bars rather than
    intraday session data.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]
    return pv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def volume_flow_indicator(
    df: pd.DataFrame,
    period: int = 130,
    coef: float = 0.2,
    vcoef: float = 2.5,
    smooth: int = 3,
    signal: int = 6,
    stdev_window: int = 30,
) -> pd.DataFrame:
    """Volume Flow Indicator (Markos Katsanos, TASC June 2004).

    On Balance Volume with two corrections, both of which matter more than
    the cumulative-sum idea they wrap:

    - a *volatility-scaled deadband* (`cutoff`): a bar only contributes
      flow if its typical-price change clears `coef` standard deviations
      of recent log returns. OBV counts every bar's full volume no matter
      how small the move, so on a quiet range it accumulates noise with a
      sign decided by rounding; here those bars contribute exactly zero.
    - a *volume cap* (`vmax`): a bar's volume counts for at most `vcoef`
      times the trailing average, so one earnings gap or index-rebalance
      print can't permanently re-base the line the way it does in OBV.

    Dividing the rolling sum by that same trailing average volume is the
    third difference: the output is in "days of average volume" rather
    than OBV's unit-less running total, so it is comparable across
    tickers and across time for one ticker, and a fixed zero line
    actually means something.

    Defaults are Katsanos's daily-chart values. For 5-15 minute bars he
    uses coef=0.1 and vcoef=3.5 -- intraday moves are smaller relative to
    their own volatility and intraday volume is spikier.

    Returns `vfi` (the smoothed line), `signal` (its own EMA), and
    `histogram` (the difference) -- the same three-column shape as
    macd(), since it is read the same way: zero-line state, signal-line
    crosses, and divergence against price.

    Needs `period + stdev_window` bars (160 by default) before it emits
    anything; every column is NaN until then rather than a partial value
    computed from a short window, since a sum over 40 bars divided by a
    40-bar average volume is not a smaller version of the same reading,
    it is a different indicator.
    """
    close = df["Close"]
    typical = (df["High"] + df["Low"] + close) / 3

    # Volatility is measured on LOG returns, then re-scaled by price to
    # get a cutoff in price units -- so the deadband is a constant
    # fraction of typical move size regardless of whether the ticker
    # trades at $3 or $900.
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(stdev_window).std()
    cutoff = coef * vinter * close

    # .shift(1): the average is of volume STRICTLY BEFORE this bar. Using
    # the current bar's own volume in the average it is about to be capped
    # against (and divided by) would let a spike raise its own ceiling.
    vave = df["Volume"].rolling(period).mean().shift(1)
    vmax = vave * vcoef
    vc = df["Volume"].clip(upper=vmax)

    money_flow = typical - typical.shift(1)
    vcp = pd.Series(np.nan, index=df.index)
    valid = cutoff.notna() & money_flow.notna()
    # Only the sign of the price change picks the sign of the flow --
    # volume never decides direction, it only scales magnitude. This is
    # not order flow and says nothing about who was buying.
    vcp[valid & (money_flow > cutoff)] = vc[valid & (money_flow > cutoff)]
    vcp[valid & (money_flow < -cutoff)] = -vc[valid & (money_flow < -cutoff)]
    vcp[valid & (money_flow.abs() <= cutoff)] = 0.0

    raw = vcp.rolling(period).sum() / vave.replace(0, np.nan)
    vfi = ema(raw, smooth)
    signal_line = ema(vfi, signal)
    return pd.DataFrame({"vfi": vfi, "signal": signal_line, "histogram": vfi - signal_line})


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of series vs. time index, computed per rolling window."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(((x - x_mean) * (y - y.mean())).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)


def curvature_r2_gain(close: pd.Series, window: int) -> pd.Series:
    """How much a quadratic fit improves on a linear fit (R^2 gain) per window.

    A high gain means the recent price path is bowing away from a straight
    line -- i.e. accelerating -- rather than moving at a steady rate. Used
    to distinguish a parabolic move from a steady trend.
    """
    x = np.arange(window, dtype=float)

    def _r2_gain(y: np.ndarray) -> float:
        if np.allclose(y, y[0]):
            return 0.0
        ss_tot = ((y - y.mean()) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        lin_fit = np.polyval(np.polyfit(x, y, 1), x)
        quad_fit = np.polyval(np.polyfit(x, y, 2), x)
        r2_lin = 1 - ((y - lin_fit) ** 2).sum() / ss_tot
        r2_quad = 1 - ((y - quad_fit) ** 2).sum() / ss_tot
        return max(0.0, r2_quad - r2_lin)

    return close.rolling(window).apply(_r2_gain, raw=True)
