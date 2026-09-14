"""Position sizing must change the STAKE and nothing else.

The failure this guards against is subtle and would not raise: if sizing
ever altered which trades were taken or when they closed, every comparison
against the unsized baseline would silently become a comparison between two
different samples -- which is exactly the trap that makes entry filters look
better than they are.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.backtest import run_backtest, vol_target_sizes
from smc_regime.regime_backtest import _ticker_compounded_return_pct, _ticker_max_drawdown_pct

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


# A calm first half and a wild second half, so the vol estimate has to move.
rng = np.random.default_rng(0)
n = 400
steps = np.concatenate([rng.normal(0, 0.005, n // 2), rng.normal(0, 0.04, n // 2)])
close = pd.Series(100 * np.cumprod(1 + steps), index=pd.date_range("2020-01-01", periods=n, freq="B"))
df = pd.DataFrame({"Close": close, "Low": close * 0.98, "High": close * 1.02, "Open": close})

# Alternating in/out so there are plenty of trades in both halves.
entry = pd.Series(False, index=df.index)
exit_ = pd.Series(False, index=df.index)
entry.iloc[::20] = True
exit_.iloc[10::20] = True
signals = pd.DataFrame({"entry": entry, "exit": exit_})

sizes = vol_target_sizes(df, target_vol_pct=20)
plain = run_backtest(df, signals)
sized = run_backtest(df, signals, size_series=sizes)

check(
    "1. sizing leaves the trade set identical -- same count",
    len(plain) == len(sized) and len(plain) > 5,
)

check(
    "2. every trade has the same entry, exit and prices",
    all(
        a.entry_date == b.entry_date and a.exit_date == b.exit_date
        and a.entry_price == b.entry_price and a.exit_price == b.exit_price
        for a, b in zip(plain, sized)
    ),
)

check(
    "3. return_pct is the asset's move and is NOT rescaled by size",
    all(a.return_pct == b.return_pct for a, b in zip(plain, sized)),
)

check(
    "4. weighted_return_pct is return_pct times size",
    all(abs(t.weighted_return_pct - t.return_pct * t.size) < 1e-9 for t in sized),
)

# The whole point: the volatile half must be held smaller than the calm half.
calm = [t.size for t in sized if t.entry_date < df.index[n // 2]]
wild = [t.size for t in sized if t.entry_date >= df.index[n // 2]]
check(
    "5. positions shrink when volatility rises",
    bool(calm) and bool(wild) and np.mean(wild) < np.mean(calm),
)

check(
    "6. size never exceeds the cap, so sizing cannot lever a position up",
    all(t.size <= 1.0 for t in sized) and sizes.max() <= 1.0,
)

check(
    "7. an unsized run is full size, so the default path is unchanged",
    all(t.size == 1.0 for t in plain),
)

# Equity maths: the helper must weight, and must reduce to the old numbers
# when no size column is present.
frame_plain = pd.DataFrame(
    {"entry_date": [t.entry_date for t in plain], "return_pct": [t.return_pct for t in plain]}
)
frame_sized = frame_plain.assign(size=[t.size for t in sized])
check(
    "8. a frame with no size column compounds exactly as before",
    _ticker_compounded_return_pct(frame_plain)
    == _ticker_compounded_return_pct(frame_plain.assign(size=1.0)),
)

check(
    "9. a null size counts as a full position, not as zero",
    _ticker_compounded_return_pct(frame_plain.assign(size=None))
    == _ticker_compounded_return_pct(frame_plain),
)

check(
    "10. sizing reduces the drawdown on this volatile series",
    _ticker_max_drawdown_pct(frame_sized) > _ticker_max_drawdown_pct(frame_plain),
)

# vol_target_sizes must not read the future: truncating the series must not
# change the values that remain.
half = vol_target_sizes(df.iloc[: n // 2 + 30], target_vol_pct=20)
check(
    "11. the size at a bar uses no data from after it",
    np.allclose(half.values, sizes.iloc[: len(half)].values, equal_nan=True),
)

def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


# A zero or negative target would divide to zero or flip the sign of every
# position -- silently, and only visible as a wrong equity curve later.
check(
    "12. a non-positive vol target is rejected rather than silently inverted",
    raises(lambda: vol_target_sizes(df, 0)) and raises(lambda: vol_target_sizes(df, -5)),
)

# Annualising a volatility scales by sqrt(bars per year), so one constant
# can only ever be right for one bar size. Using the daily 252 on hourly
# data understates vol 2.55x and drives almost every position to the cap.
hourly = vol_target_sizes(df, target_vol_pct=20, interval="1h")
weekly = vol_target_sizes(df, target_vol_pct=20, interval="1w")
daily = vol_target_sizes(df, target_vol_pct=20, interval="1d")

check(
    "13. a finer interval annualises to a HIGHER vol, so smaller positions",
    hourly.dropna().mean() < daily.dropna().mean(),
)

check(
    "14. a coarser interval annualises to a LOWER vol, so larger positions",
    weekly.dropna().mean() > daily.dropna().mean(),
)

# The ratio is exactly sqrt(bars_per_year) where neither is at the cap.
free = (daily < 0.999) & (hourly < 0.999)
if free.any():
    ratio = (daily[free] / hourly[free]).mean()
    expected = np.sqrt((252 * 6.5) / 252)
    check(
        "15. the size ratio between intervals is exactly sqrt(bars per year)",
        abs(ratio - expected) < 0.01,
    )
else:
    check("15. the size ratio between intervals is exactly sqrt(bars per year)", False)

check(
    "16. an unknown interval falls back to the daily factor rather than raising",
    vol_target_sizes(df, 20, interval="nonsense").equals(daily),
)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
