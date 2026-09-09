"""Checks for smc_regime.portfolio: equity-curve construction and the
risk-adjusted statistics built on it.

Synthetic prices and hand-built trades throughout -- no Tiingo key needed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_regime.portfolio import (
    TRADING_DAYS,
    hedged_returns,
    market_model,
    performance_stats,
    portfolio_returns,
    trade_daily_returns,
)


@dataclass
class FakeTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp


idx = pd.bdate_range("2024-01-01", periods=10)
# +1% every day, so every held day's return is exactly 0.01
prices = pd.Series(100 * 1.01 ** np.arange(10), index=idx)
df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": 1e6})

# 1. A trade entered at close of day 1 and exited at close of day 4 earns the
#    returns of days 2, 3 and 4 -- not day 1, which is already priced in.
trade = FakeTrade(idx[1], idx[4])
r = trade_daily_returns(df, [trade])
assert r.notna().sum() == 3, r.notna().sum()
assert list(r.dropna().index) == list(idx[2:5])
assert np.allclose(r.dropna().to_numpy(), 0.01)
print("1. held days are entry+1 .. exit  OK")

# 2. Costs are charged once at entry and once at exit, so a 10bp-per-side
#    round trip removes 20bp from the trade in total.
net = trade_daily_returns(df, [trade], cost_bps_per_side=10.0)
assert np.isclose((net.dropna() - r.dropna()).sum(), -0.002), (net.dropna() - r.dropna()).sum()
assert np.isclose(net.dropna().iloc[1], 0.01), "a middle day should carry no cost"
print("2. costs charged at entry and exit only  OK")

# 3. A short earns the negative of the price return.
short = trade_daily_returns(df, [trade], side="short")
assert np.allclose(short.dropna().to_numpy(), -0.01)
print("3. short side flips the sign  OK")

# 4. Equal weight across whatever is open; cash when nothing is.
a = pd.Series([np.nan, 0.02, 0.02, np.nan], index=idx[:4])
b = pd.Series([np.nan, 0.00, np.nan, np.nan], index=idx[:4])
port = portfolio_returns({"A": a, "B": b}, rf_annual=0.0)
assert list(port["open_positions"]) == [0, 2, 1, 0]
assert np.isclose(port["return"].iloc[1], 0.01)   # mean of 2% and 0%
assert np.isclose(port["return"].iloc[2], 0.02)   # only A open
assert np.isclose(port["return"].iloc[0], 0.0)    # flat -> cash at rf=0
assert list(port["invested"]) == [False, True, True, False]
print("4. equal weight across open positions, cash when flat  OK")

# 5. Cash days earn the risk-free rate, not zero.
port_rf = portfolio_returns({"A": a}, rf_annual=0.05)
expected_daily = (1.05) ** (1 / TRADING_DAYS) - 1
assert np.isclose(port_rf["return"].iloc[0], expected_daily)
print("5. flat days earn the risk-free rate  OK")

# 6. Sharpe matches the definition computed independently.
rng = np.random.default_rng(0)
series = pd.Series(rng.normal(0.0005, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
stats = performance_stats(series, rf_annual=0.0)
expected_sharpe = series.mean() / series.std(ddof=1) * np.sqrt(TRADING_DAYS)
assert np.isclose(stats["sharpe"], expected_sharpe), (stats["sharpe"], expected_sharpe)
print("6. Sharpe matches its definition  OK")

# 7. With the standard full-sample denominator, downside deviation can never
#    exceed total volatility, so Sortino >= Sharpe for ANY positive-mean
#    series -- "Sortino beat Sharpe" is not itself evidence of anything. What
#    carries the skew information is the RATIO between them: a series whose
#    volatility is mostly upside scores far higher on Sortino than one whose
#    volatility is mostly drawdown.
days = pd.bdate_range("2020-01-01", periods=100)
right = pd.Series([-0.005] * 90 + [0.20] * 10, index=days)   # small bleeds, rare big wins
left = pd.Series([0.01] * 95 + [-0.10] * 5, index=days)      # steady gains, rare big losses
right_ratio = performance_stats(right)["sortino"] / performance_stats(right)["sharpe"]
left_ratio = performance_stats(left)["sortino"] / performance_stats(left)["sharpe"]
assert right_ratio > 10, right_ratio
assert left_ratio < 1.2, left_ratio
assert right_ratio > left_ratio
print("7. Sortino/Sharpe ratio separates upside skew from downside  OK")

# 8. Sortino is finite only when there are losing days at all.
allup = pd.Series([0.01] * 50, index=pd.bdate_range("2020-01-01", periods=50))
assert np.isnan(performance_stats(allup)["sortino"])
print("8. no downside -> Sortino is NaN, not infinity  OK")

# 9. THE HEDGE IS SIGN-AWARE. A market-neutral short book is short the name
#    and LONG the benchmark, so the benchmark is added back, not subtracted.
#    Subtracting on both sides double-counts the market against the shorts --
#    this test exists because that bug was live once.
bench = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx[:4])
book = pd.DataFrame({"return": [-0.005, -0.005, -0.005, -0.005], "open_positions": [1, 1, 1, 1],
                     "invested": [True] * 4}, index=idx[:4])
long_hedged = hedged_returns(book, bench, side="long")
short_hedged = hedged_returns(book, bench, side="short")
assert np.allclose(long_hedged.to_numpy(), -0.015)    # r - b
assert np.allclose(short_hedged.to_numpy(), 0.005)    # r + b
assert not np.allclose(short_hedged.to_numpy(), long_hedged.to_numpy())
print("9. hedge adds the benchmark for shorts, subtracts for longs  OK")

# 10. Flat days contribute no hedged return -- you are not short the
#     benchmark while sitting in cash.
book_flat = book.copy()
book_flat.loc[idx[0], "invested"] = False
assert hedged_returns(book_flat, bench, "long").iloc[0] == 0.0
print("10. cash days carry no hedge  OK")

# 11. market_model recovers a known beta and alpha.
bench_r = pd.Series(rng.normal(0.0004, 0.009, 1000), index=series.index)
strategy = 1.5 * bench_r + 0.0002          # beta 1.5, 2bp/day alpha, no noise
mm = market_model(strategy, bench_r, rf_annual=0.0)
assert np.isclose(mm["beta"], 1.5), mm["beta"]
assert np.isclose(mm["alpha_ann_pct"], ((1.0002) ** TRADING_DAYS - 1) * 100), mm["alpha_ann_pct"]
print("11. market_model recovers beta and alpha  OK")

print("\nall portfolio checks passed")
