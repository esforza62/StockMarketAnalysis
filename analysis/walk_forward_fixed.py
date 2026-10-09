"""Walk-forward with the configuration FIXED, no selection step.

The selected walk-forward kept landing on 10 slots by fallback, because a
rule that maximises in-sample CAGR always reaches for concentration -- and
10 slots is the widest-spread, deepest-drawdown cell in the grid. That
measures the selection rule, not the strategy.

Here nothing is selected. Every (strategy, slots) cell is simply reported
on every out-of-sample fold, so there is no in-sample step to overfit and
no look-ahead to worry about. The question it answers: held at a fixed
configuration, does this beat buying the index, fold by fold?
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime.portfolio import simulate_seeds
from smc_regime.portfolio_cli import split_trades

# Where the cached universe/trade pickles live. Defaults beside this file so
# a fresh checkout works; override with SMC_CACHE to reuse an existing cache.
S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
ALL = pickle.load(open(f"{S}/wf_trades.pkl", "rb"))
TZ = prices["SPY"].index.tz
spy = prices["SPY"]["Close"]
sma200 = spy.rolling(200).mean()
BOUNDS = [f"{y}-01-01" for y in (2022, 2023, 2024, 2025, 2026)]
SLOTS = (10, 20, 40)
COSTS = (0.0, 0.05, 0.10)
SEEDS = 8

rows = []
for i, b in enumerate(BOUNDS):
    lo = pd.Timestamp(b, tz=TZ)
    hi = pd.Timestamp(BOUNDS[i + 1], tz=TZ) if i + 1 < len(BOUNDS) else spy.index[-1] + pd.Timedelta(days=1)
    s_oos = spy[(spy.index >= lo) & (spy.index < hi)]
    if len(s_oos) < 60:
        continue
    yrs = (s_oos.index[-1] - s_oos.index[0]).days / 365.25
    spy_c = ((s_oos.iloc[-1] / s_oos.iloc[0]) ** (1 / yrs) - 1) * 100
    spy_dd = float(((s_oos / s_oos.cummax()) - 1).min() * 100)
    above = float((s_oos > sma200.reindex(s_oos.index)).mean() * 100)
    p_oos = {t: d[(d.index >= lo) & (d.index < hi)] for t, d in prices.items()}
    t0 = time.time()
    print(f"\n=== OOS {b} -> {hi.date()}  ({yrs:.2f}y)  SPY {spy_c:+.2f}%  "
          f"maxDD {spy_dd:.1f}%  days above 200sma {above:.0f}% ===", flush=True)
    for strat, trs_by_tk in ALL.items():
        # trades ENTERING inside the window and exiting before it ends;
        # straddlers dropped, same rule as the single split
        tr = {}
        for tk, trs in trs_by_tk.items():
            inside = [t for t in trs if t.entry_date >= lo]
            tr[tk], _, _ = split_trades(inside, hi)
        for n in SLOTS:
            line = f"  {strat:16s} {n:3d} slots |"
            for bp in COSTS:
                sw = simulate_seeds(tr, p_oos, seeds=SEEDS, max_positions=n, slippage_pct=bp)
                c, d = sw.spread("cagr_pct"), sw.spread("max_drawdown_pct")
                rows.append({"fold": b, "strategy": strat, "slots": n, "bp": bp * 100,
                             "cagr": c["mean"], "sd": c["sd"], "min": c["min"], "max": c["max"],
                             "maxdd": d["mean"], "spy_cagr": spy_c, "spy_dd": spy_dd,
                             "above200": above,
                             "beat": "yes" if c["min"] > spy_c else "mixed" if c["max"] > spy_c else "no",
                             "rej": float(np.mean([r.rejection_rate for r in sw.runs]) * 100)})
                line += f" {int(bp*100)}bp {c['mean']:+7.2f}+/-{c['sd']:4.2f}"
            print(line, flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

pd.DataFrame(rows).to_csv(f"{S}/wf40_results.csv", index=False)
print(f"\nwrote {S}/wf40_results.csv")
