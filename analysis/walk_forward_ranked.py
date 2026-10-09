"""Ranked selection and a quality floor, walk-forward.

Two things the random-tie-break book could not do:

  RANK -- 64-89% of signals were dropped by coin flip. RSI(2) has a DEPTH
  (a name at RSI 2 is not a name at RSI 9.9) and that ordering is causal
  and already in hand, so scarce slots can go to the deepest signals.

  HOLD CASH -- the book ran 99.9% invested with drawdown equal to SPY's,
  which is the signature of owning the index. RSI(2)'s appeal on SPY alone
  was 83% of the return for 28% exposure. A floor restores the ability to
  sit out: a slot stays empty unless the signal clears a depth bar.

score = 10 - RSI(2) at the entry bar, so higher is deeper. Entries require
RSI < 10, so depth lands in (0, 10]. floor 5 means RSI < 5; floor 8 means
RSI < 2. Nothing here is selected on in-sample data -- every cell is simply
reported on every fold, so there is no rule to overfit. Picking a floor on
these numbers would need its own holdout, and the honest guard is whether a
floor helps CONSISTENTLY across folds rather than on average.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime import indicators as ind
from smc_regime.portfolio import simulate, simulate_seeds
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
SLOTS = (20, 40)
FLOORS = (None, 3.0, 5.0, 7.0, 8.0)
COSTS = (0.05, 0.10)
SEEDS = 8

print("computing RSI(2) depth scores...", flush=True)
SCORES = {}
for tk, df in prices.items():
    r = ind.rsi(df["Close"], 2)
    for d, v in r.items():
        if pd.notna(v):
            SCORES[(tk, d)] = 10.0 - float(v)
print(f"  {len(SCORES)} scored bars\n", flush=True)

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
    print(f"=== OOS {b} -> {hi.date()}  SPY {spy_c:+.2f}%  dd {spy_dd:.1f}%  "
          f"trend {above:.0f}% ===", flush=True)
    for strat, trs_by_tk in ALL.items():
        tr = {}
        for tk, trs in trs_by_tk.items():
            tr[tk], _, _ = split_trades([t for t in trs if t.entry_date >= lo], hi)
        for n in SLOTS:
            t0 = time.time()
            # random baseline, seed-averaged -- what the book did before
            for bp in COSTS:
                sw = simulate_seeds(tr, p_oos, seeds=SEEDS, max_positions=n, slippage_pct=bp)
                c = sw.spread("cagr_pct"); d = sw.spread("max_drawdown_pct")
                rows.append(dict(fold=b, strategy=strat, slots=n, mode="random", floor=np.nan,
                                 bp=bp*100, cagr=c["mean"], sd=c["sd"], lo=c["min"], hi=c["max"],
                                 maxdd=d["mean"],
                                 fill=float(np.mean([r.stats["slot_fill_pct"] for r in sw.runs])),
                                 deployed=float(np.mean([r.stats["capital_deployed_pct"] for r in sw.runs])),
                                 declined=0, spy_cagr=spy_c, spy_dd=spy_dd, trend=above))
            # ranked, with and without a floor -- deterministic, so no seeds
            for fl in FLOORS:
                for bp in COSTS:
                    r1 = simulate(tr, p_oos, max_positions=n, slippage_pct=bp,
                                  selector="ranked", scores=SCORES, score_floor=fl)
                    rows.append(dict(fold=b, strategy=strat, slots=n,
                                     mode="ranked" if fl is None else "ranked+floor",
                                     floor=np.nan if fl is None else fl, bp=bp*100,
                                     cagr=r1.stats["cagr_pct"], sd=0.0,
                                     lo=r1.stats["cagr_pct"], hi=r1.stats["cagr_pct"],
                                     maxdd=r1.stats["max_drawdown_pct"],
                                     fill=r1.stats["slot_fill_pct"],
                                     deployed=r1.stats["capital_deployed_pct"],
                                     declined=r1.declined, spy_cagr=spy_c, spy_dd=spy_dd,
                                     trend=above))
            sub = [x for x in rows if x["fold"] == b and x["strategy"] == strat
                   and x["slots"] == n and x["bp"] == 5.0]
            msg = "  ".join(f"{x['mode'][:6]}{'' if np.isnan(x['floor']) else int(x['floor'])}"
                            f" {x['cagr']:+6.2f}({x['fill']:.0f}%)" for x in sub)
            print(f"  {strat:16s}{n:3d}sl 5bp | {msg}  [{time.time()-t0:.0f}s]", flush=True)
    print(flush=True)

pd.DataFrame(rows).to_csv(f"{S}/wfrank_results.csv", index=False)
print(f"wrote {S}/wfrank_results.csv")
