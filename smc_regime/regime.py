"""Rule-based classification of price-movement regime: choppy, trending, or parabolic."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind


@dataclass
class RegimeThresholds:
    er_window: int = 14
    er_trend_min: float = 0.30

    adx_window: int = 14
    adx_trend_min: float = 20.0

    atr_window: int = 14
    atr_roc_window: int = 5
    atr_roc_parabolic_min: float = 0.05  # ATR% must jump >=5% over atr_roc_window bars

    ema_window: int = 20
    extension_parabolic_min: float = 3.0  # price >= N ATRs away from its EMA

    curvature_window: int = 14
    # curvature_r2_gain is reported as a diagnostic column but is NOT part of
    # the parabolic gate below: at realistic daily move sizes and a 14-bar
    # window, even a clean exponential move only produces an R^2 gain of
    # ~0.006 vs. a straight line (a line already explains most of the local
    # variance), and that's within the noise range of ordinary choppy/trending
    # bars too -- it doesn't reliably discriminate at this window length.


def compute_regime_features(df: pd.DataFrame, t: RegimeThresholds = RegimeThresholds()) -> pd.DataFrame:
    close = df["Close"]

    out = pd.DataFrame(index=df.index)
    out["efficiency_ratio"] = ind.efficiency_ratio(close, t.er_window)
    out["adx"] = ind.adx(df, t.adx_window)
    out["atr_pct"] = ind.atr_pct(df, t.atr_window)
    out["atr_pct_roc"] = out["atr_pct"].pct_change(t.atr_roc_window)
    out["ema_extension_atr"] = ind.ema_extension_atr_units(df, t.ema_window, t.atr_window)
    out["curvature_r2_gain"] = ind.curvature_r2_gain(close, t.curvature_window)
    out["slope"] = ind.rolling_slope(close, t.er_window)
    return out


def classify_regime(df: pd.DataFrame, t: RegimeThresholds = RegimeThresholds()) -> pd.DataFrame:
    """Label each bar 'choppy', 'trending', or 'parabolic', plus direction.

    Rule-based and threshold-driven on purpose: it's meant to be inspected
    and tuned against real charts before anything model-based gets layered
    on top of it.

    A bar is:
    - trending  if Efficiency Ratio or ADX clears its threshold
    - parabolic if it's trending AND price is stretched far from its EMA
                (in ATR units) AND volatility (ATR%) is itself accelerating
    - choppy    otherwise

    `curvature_r2_gain` is also reported per bar as a diagnostic (how much a
    quadratic fit improves on a linear one) but isn't used as a gate -- see
    the note on `RegimeThresholds.curvature_window`.
    """
    feats = compute_regime_features(df, t)

    is_trending = (feats["efficiency_ratio"] >= t.er_trend_min) | (feats["adx"] >= t.adx_trend_min)

    is_parabolic = (
        is_trending
        & (feats["ema_extension_atr"].abs() >= t.extension_parabolic_min)
        & (feats["atr_pct_roc"] >= t.atr_roc_parabolic_min)
    )

    regime = pd.Series("choppy", index=df.index)
    regime[is_trending] = "trending"
    regime[is_parabolic] = "parabolic"

    direction = pd.Series("flat", index=df.index)
    direction[feats["slope"] > 0] = "up"
    direction[feats["slope"] < 0] = "down"
    direction[regime == "choppy"] = "n/a"

    result = feats.copy()
    result["regime"] = regime
    result["direction"] = direction
    return result
