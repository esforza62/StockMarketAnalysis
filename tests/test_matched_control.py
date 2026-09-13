"""The matched control has to be right about the boring case first.

Its whole job is subtracting "what you'd have earned entering at random in
the same regime" from a strategy's return. If that subtraction is off by a
bar, every edge figure it produces is wrong in the same direction and
nothing about the output would look suspicious -- so the checks here are
about alignment, not about any strategy.

Bars are synthetic -- these run without a Tiingo key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import Trade
from smc_regime.matched_control import control_returns, summarize_edge


def constant_growth_bars(n: int = 200, daily: float = 0.01) -> pd.DataFrame:
    closes = 100 * (1 + daily) ** np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000_000.0},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def one_bucket(index: pd.Index, regime: str = "trending", direction: str = "up") -> pd.DataFrame:
    return pd.DataFrame({"regime": regime, "direction": direction}, index=index)


def test_constant_growth_has_exactly_zero_edge() -> None:
    """On a series compounding at a fixed rate, every H-bar window returns
    the same thing, so the control must equal each trade's return exactly
    -- whatever the entries were. Any edge here is an alignment bug."""
    bars = constant_growth_bars()
    regime = one_bucket(bars.index)
    trades = [
        Trade(bars.index[10], bars.index[25], bars["Close"].iloc[10], bars["Close"].iloc[25]),
        Trade(bars.index[60], bars.index[68], bars["Close"].iloc[60], bars["Close"].iloc[68]),
        Trade(bars.index[100], bars.index[140], bars["Close"].iloc[100], bars["Close"].iloc[140]),
    ]
    out = control_returns(bars, regime, trades)
    assert len(out) == 3, f"expected 3 comparable trades, got {len(out)}"
    assert np.allclose(out["edge_pct"], 0.0, atol=1e-9), f"non-zero edge on a constant series: {out['edge_pct'].tolist()}"


def test_a_better_exit_shows_up_as_positive_edge() -> None:
    """Same entry, an exit priced above what the series actually did: the
    edge must be positive and equal to the gap, so the sign convention is
    (strategy - control), not the reverse."""
    bars = constant_growth_bars()
    regime = one_bucket(bars.index)
    entry_price = bars["Close"].iloc[10]
    honest = Trade(bars.index[10], bars.index[20], entry_price, bars["Close"].iloc[20])
    lucky = Trade(bars.index[10], bars.index[20], entry_price, bars["Close"].iloc[20] * 1.05)

    honest_edge = control_returns(bars, regime, [honest])["edge_pct"].iloc[0]
    lucky_edge = control_returns(bars, regime, [lucky])["edge_pct"].iloc[0]
    assert np.isclose(honest_edge, 0.0, atol=1e-9)
    assert lucky_edge > honest_edge, "a better exit did not register as positive edge"


def test_control_is_drawn_only_from_the_trade_s_own_regime() -> None:
    """A trade entered in one regime must not be compared against bars from
    another.

    The regime label flips at bar 100 but the price behaviour only changes
    at bar 110, so every trending-bucket start bar still has its full
    10-bar horizon inside the growing stretch. The flat bars are therefore
    excluded by their LABEL alone -- if the control ignored the label and
    swept the whole series, the flat tail would drag it down and invent
    positive edge out of nothing.
    """
    n, horizon = 200, 10
    growth_bars = 110
    closes = np.concatenate([
        100 * 1.01 ** np.arange(growth_bars),
        np.full(n - growth_bars, 100 * 1.01 ** (growth_bars - 1)),
    ])
    bars = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000_000.0},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    regime = pd.concat([one_bucket(bars.index[:100]), one_bucket(bars.index[100:], "choppy", "n/a")])

    trade = Trade(bars.index[10], bars.index[10 + horizon], bars["Close"].iloc[10], bars["Close"].iloc[10 + horizon])
    out = control_returns(bars, regime, [trade])
    assert out["bucket"].iloc[0] == "trending/up"
    assert np.isclose(out["edge_pct"].iloc[0], 0.0, atol=1e-9), (
        f"control leaked across regimes: edge {out['edge_pct'].iloc[0]}"
    )

    # And the control really is the same-regime number, not a whole-series
    # average -- sanity that the assertion above isn't passing by accident.
    whole_series = ((bars["Close"].shift(-horizon) / bars["Close"] - 1) * 100).dropna().mean()
    assert out["control_pct"].iloc[0] > whole_series


def test_summarize_respects_the_minimum_trade_count() -> None:
    bars = constant_growth_bars()
    regime = one_bucket(bars.index)
    trades = [
        Trade(bars.index[i], bars.index[i + 5], bars["Close"].iloc[i], bars["Close"].iloc[i + 5])
        for i in range(0, 40, 5)
    ]
    out = control_returns(bars, regime, trades)
    assert summarize_edge(out, min_trades=30).empty, "a thin bucket was reported anyway"
    kept = summarize_edge(out, min_trades=5)
    assert len(kept) == 1 and kept["trade_count"].iloc[0] == len(trades)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok  {name}")
    print("\nall matched-control tests passed")
