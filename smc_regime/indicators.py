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
    return dx.ewm(alpha=1 / window, adjust=False).mean()


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


def ema_extension_atr_units(df: pd.DataFrame, ema_window: int = 20, atr_window: int = 14) -> pd.Series:
    """Signed distance of close from its EMA, expressed in ATR units."""
    close = df["Close"]
    dist = close - ema(close, ema_window)
    return dist / atr(df, atr_window).replace(0, np.nan)


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
