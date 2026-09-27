"""Contamination guards in smc_regime.split_guard, on constructed series.

Unadjusted intraday data carries fabricated returns from three sources: a
split, a ticker whose symbol changed hands, and a bad tick. The split cache
only knows about the first. These tests cover the discontinuity guard that
catches the other two, because a fabricated return survives averaging and
nothing downstream flags it -- one BNY trade at +1283% was enough to make
the worst bucket of a 2,690-trade summary look like the best.
"""
import pandas as pd

from smc_regime.split_guard import (
    filter_contaminated_trades,
    filter_discontinuity_trades,
    price_discontinuities,
)

index = pd.date_range("2024-01-01 09:30", periods=10, freq="1h", tz="UTC")
closes = [100.0, 101.0, 100.5, 101.5, 10.2, 10.3, 10.1, 10.4, 10.2, 10.5]
df = pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                   "Volume": [1] * 10}, index=index)

# The 101.5 -> 10.2 step is the only implausible one; ordinary 1% wiggles
# on either side of it are not.
jumps = price_discontinuities(df)
assert jumps == [index[4]], jumps

# A calmer series has none at all, and the threshold is adjustable.
calm = df.copy()
calm[["Open", "High", "Low", "Close"]] = 100.0
assert price_discontinuities(calm) == []
assert price_discontinuities(df, max_bar_move=0.99) == []

trades = pd.DataFrame(
    {
        "ticker": ["X", "X", "X"],
        "entry_date": [index[0], index[5], index[2]],
        "exit_date": [index[3], index[8], index[6]],  # before / after / straddling
        "return_pct": [1.5, 2.0, -89.9],
    }
)
kept = filter_discontinuity_trades(trades, jumps)
assert list(kept["return_pct"]) == [1.5, 2.0], list(kept["return_pct"])

# A trade ending exactly ON the jump bar is already contaminated -- that bar's
# close is the fabricated price the trade would exit at.
touching = pd.DataFrame({"ticker": ["X"], "entry_date": [index[2]],
                         "exit_date": [index[4]], "return_pct": [-89.9]})
assert filter_discontinuity_trades(touching, jumps).empty

# No jumps, or no trades, changes nothing.
assert len(filter_discontinuity_trades(trades, [])) == 3
assert filter_discontinuity_trades(trades.iloc[:0], jumps).empty

# The two guards are independent. This split falls in the flat stretch AFTER
# the discontinuity, so the jump-based guard keeps that trade and only the
# cache catches it -- which is the case for an adjusted series, or a split
# whose ratio is too small to look like a jump.
by_cache = filter_contaminated_trades(trades, {"X": [index[6]]})
assert list(by_cache["return_pct"]) == [1.5, -89.9], list(by_cache["return_pct"])
assert 2.0 in list(filter_discontinuity_trades(trades, jumps)["return_pct"])

# And a cache holding some other ticker's splits leaves these alone.
assert len(filter_contaminated_trades(trades, {"OTHER": [index[6]]})) == 3

print("all split guard tests passed")
