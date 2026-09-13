"""Volume Flow Indicator: the properties that separate it from OBV.

VFI is only worth having instead of OBV because of three specific
mechanics -- a volatility deadband, a volume cap, and normalization by
average volume. Each is a line of code that would still produce a
plausible-looking oscillator if it were wrong, so each is checked here
directly rather than through a strategy's trade count.

The no-lookahead check is the important one: `vave` is a trailing average
that must be shifted off the current bar, and a `.rolling()` window that
silently became centered, or a cap computed against an average that
includes the bar being capped, would both leak future information into a
backtest and flatter every result downstream.

Bars are synthetic -- these run without a Tiingo key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import indicators as ind
from smc_regime.strategies import STRATEGIES

PERIOD = 130
STDEV_WINDOW = 30
WARMUP = PERIOD + STDEV_WINDOW


def make_bars(closes: np.ndarray, volume: float = 1_000_000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": np.full(len(closes), volume, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=len(closes), freq="B"),
    )


def drifting_bars(n: int, drift: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return make_bars(100 * np.cumprod(1 + rng.normal(drift, 0.01, n)))


def test_deadband_zeroes_out_a_flat_series() -> None:
    """A series that never moves has no flow at all -- not a small
    residual. OBV would bounce around on tie-breaking; the cutoff makes
    every bar contribute exactly 0."""
    vfi = ind.volume_flow_indicator(make_bars(np.full(250, 100.0)))["vfi"]
    values = vfi.dropna()
    assert not values.empty, "flat series produced no VFI values at all"
    assert np.allclose(values, 0.0), f"flat series moved VFI: max |vfi| = {values.abs().max()}"


def test_warmup_is_fully_masked() -> None:
    """Nothing before period + stdev_window bars, and something after --
    a partial sum over a short window is a different indicator, not an
    early reading of this one."""
    vfi = ind.volume_flow_indicator(drifting_bars(300, 0.004))["vfi"]
    assert vfi.iloc[: WARMUP - 1].isna().all(), "VFI emitted a value before its window filled"
    assert vfi.iloc[WARMUP:].notna().all(), "VFI still NaN after its window filled"


def test_sign_follows_net_flow_direction() -> None:
    """Sustained gains read as accumulation, sustained losses as
    distribution. Volume is constant in both, so only the price-change
    sign can be driving this."""
    assert ind.volume_flow_indicator(drifting_bars(300, 0.004))["vfi"].iloc[-1] > 0
    assert ind.volume_flow_indicator(drifting_bars(300, -0.004))["vfi"].iloc[-1] < 0


def test_volume_cap_bounds_a_spike() -> None:
    """A 500x volume bar must count for exactly vcoef times average
    volume and no more -- so it produces the identical VFI to a bar whose
    volume really was vcoef x average.

    The spike goes on the LAST bar on purpose: anywhere earlier it would
    also shift the trailing average for every later bar, and the two
    series would then differ downstream for a legitimate reason, which
    would make this assertion test nothing.
    """
    bars = drifting_bars(300, 0.004)
    trailing_average = bars["Volume"].rolling(PERIOD).mean().shift(1).iloc[-1]
    volume_col = bars.columns.get_loc("Volume")

    spiked = bars.copy()
    spiked.iloc[-1, volume_col] = 500_000_000.0
    at_the_cap = bars.copy()
    at_the_cap.iloc[-1, volume_col] = trailing_average * 2.5

    spiked_vfi = ind.volume_flow_indicator(spiked)["vfi"].iloc[-1]
    capped_vfi = ind.volume_flow_indicator(at_the_cap)["vfi"].iloc[-1]
    assert np.isclose(spiked_vfi, capped_vfi), f"spike escaped the cap: {spiked_vfi} vs {capped_vfi}"

    # And the cap is a cap, not a clamp to the average: the spike bar
    # still counts for more than an ordinary one.
    assert spiked_vfi > ind.volume_flow_indicator(bars)["vfi"].iloc[-1]


def test_no_lookahead() -> None:
    """Every bar's VFI must be identical whether or not later bars exist.

    This is what a shifted trailing average buys, and the failure it
    guards is silent: an unshifted vave still produces a smooth,
    plausible line, one that has seen the future.
    """
    bars = drifting_bars(300, 0.004)
    cutoff = 220
    full = ind.volume_flow_indicator(bars)
    truncated = ind.volume_flow_indicator(bars.iloc[:cutoff])

    for column in ("vfi", "signal", "histogram"):
        a = full[column].iloc[:cutoff]
        b = truncated[column]
        assert a.isna().equals(b.isna()), f"{column}: NaN positions differ between full and truncated runs"
        assert np.allclose(a.dropna(), b.dropna()), f"{column}: values changed when later bars were removed"


def test_strategies_emit_no_signal_during_warmup() -> None:
    """A strategy reading an all-NaN indicator must produce no trades
    rather than False-y comparisons that happen to look like signals."""
    bars = drifting_bars(300, 0.004)
    for name in ("vfi_zero_cross", "vfi_signal_cross", "vfi_divergence"):
        signals = STRATEGIES[name](bars)
        assert signals.index.equals(bars.index), f"{name}: signals not aligned to the input index"
        assert signals["entry"].dtype == bool and signals["exit"].dtype == bool, f"{name}: non-boolean signals"
        warmup_signals = signals.iloc[: WARMUP - 1]
        assert not warmup_signals["entry"].any(), f"{name}: entered a trade before VFI had a value"
        assert not warmup_signals["exit"].any(), f"{name}: exited a trade before VFI had a value"


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok  {name}")
    print("\nall volume-flow tests passed")
