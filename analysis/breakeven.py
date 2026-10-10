"""Break-even transaction cost for the rsi2 variants, derived not quoted.

An earlier version of portfolio_cli's docstring claimed "break-even at
40.8bp per side". That was a ROUND-TRIP figure carrying a per-side label,
overstating the cost tolerance by 2.2x, and it sat in the same docstring as
walk-forward results that contradicted it (beats SPY at 0bp, mixed at 5bp,
loses every fold at 10bp). This recomputes it from the trades so the number
is checkable.

A basis point is 0.01%. "Per side" means charged on each fill, so a round
trip pays twice: break-even per side is half the per-trade excess.

The baseline is a same-ticker, same-duration hold -- "does timing the entry
beat holding this name for the same number of bars". Against SPY the
tolerance is lower still, since that baseline hands over market beta with
no turnover.
"""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
ALL = pickle.load(open(f"{S}/wf_trades.pkl", "rb"))

for strat in ("rsi2_prior_high", "rsi2"):
    rets, holds, bench = [], [], []
    for tk, trs in ALL[strat].items():
        df = prices.get(tk)
        if df is None or len(df) < 60:
            continue
        c = df["Close"]
        pos = {d: i for i, d in enumerate(df.index)}
        cache = {}
        for t in trs:
            i, j = pos.get(t.entry_date), pos.get(t.exit_date)
            if i is None or j is None or j <= i:
                continue
            n = j - i
            rets.append((t.exit_price / t.entry_price - 1) * 100)
            holds.append(n)
            if n not in cache:
                cache[n] = float((c.shift(-n) / c - 1).mul(100).mean())
            bench.append(cache[n])
    r, b, h = np.array(rets), np.array(bench), np.array(holds)
    ex = (r - b).mean()
    # ~252 trading days a year; the book is not always in a trade, so this
    # is an upper bound on turnover per slot
    rt_per_year = 252 / h.mean()
    print(f"\n=== {strat} ===")
    print(f"  trades {len(r):,}   mean hold {h.mean():.1f} bars   "
          f"~{rt_per_year:.0f} round trips/slot/year")
    print(f"  gross {r.mean():+.4f}%   baseline {b.mean():+.4f}%   "
          f"excess {ex:+.4f}% ({ex*100:.1f}bp)")
    print(f"  break-even: {ex*100:.1f}bp round trip = {ex*100/2:.1f}bp PER SIDE")
    print(f"  {'cost/side':>10}{'net/trade':>12}{'annual drag':>14}")
    for cbp in (0, 5, 10, 20):
        print(f"  {cbp:>9}bp{ex - 2*cbp/100:>+11.4f}%{rt_per_year*2*cbp/100:>13.1f}%")
