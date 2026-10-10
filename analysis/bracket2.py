"""+10% target / -5% stop on the rsi2 variants, with the random control.

Same 2:1 ratio as 20/10, so the break-even hit rate is unchanged at 33.3%.
Half the width means both levels are far more reachable inside the
strategy's own 4-8 bar hold, so WITHIN mode should finally bind.

The random-entry control is included from the start, not bolted on. At
20/10 the pure bracket beat the strategy's own exit by +2.55pp and beat a
RANDOM entry by +0.38pp -- the rest was the market and a survivor
universe. Any result here gets the same test.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime.backtest import backtest_strategy

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
TARGET, STOP, MAXHOLD = 0.10, 0.05, 250
rng = np.random.default_rng(7)


def walk(hi, lo, op, i, entry, last):
    tgt, stp = entry * (1 + TARGET), entry * (1 - STOP)
    for k in range(i + 1, min(last, len(hi) - 1) + 1):
        if lo[k] <= stp:                       # stop first when both hit
            return min(op[k], stp) / entry - 1, "stop", k - i
        if hi[k] >= tgt:
            return max(op[k], tgt) / entry - 1, "target", k - i
    return None, "neither", min(last, len(hi) - 1) - i


rows, counts = [], {}
t0 = time.time()
for strat in ("rsi2", "rsi2_prior_high"):
    n = 0
    for tk, df in prices.items():
        if len(df) < 300:
            continue
        try:
            trades = backtest_strategy(df, strat, interval="1d")
        except Exception:
            continue
        hi, lo, op = df["High"].to_numpy(), df["Low"].to_numpy(), df["Open"].to_numpy()
        pos = {d: i for i, d in enumerate(df.index)}
        for t in trades:
            i, j = pos.get(t.entry_date), pos.get(t.exit_date)
            if i is None or j is None or j <= i:
                continue
            e = float(t.entry_price)
            base = float(t.exit_price) / e - 1
            rw, ow, _ = walk(hi, lo, op, i, e, j)
            rp, op_, bp = walk(hi, lo, op, i, e, i + MAXHOLD)
            if rp is None:
                rp = float(df["Close"].iloc[min(i + MAXHOLD, len(df) - 1)]) / e - 1
            rows.append(dict(strategy=strat, ticker=tk, entry=t.entry_date, bars=j - i,
                             baseline=base, within=base if rw is None else rw,
                             within_out=ow, pure=rp, pure_out=op_, pure_bars=bp))
            n += 1
        if strat == "rsi2":
            counts[tk] = counts.get(tk, 0) + len([t for t in trades if pos.get(t.entry_date) is not None])
    print(f"  {strat}: {n:,} trades  [{time.time()-t0:.0f}s]", flush=True)

d = pd.DataFrame(rows); d["year"] = pd.to_datetime(d["entry"]).dt.year

ctrl = []
for tk, n in counts.items():
    df = prices.get(tk)
    if df is None or len(df) < 300:
        continue
    hi, lo, op, cl = (df["High"].to_numpy(), df["Low"].to_numpy(),
                      df["Open"].to_numpy(), df["Close"].to_numpy())
    for i in rng.integers(0, max(1, len(df) - 2), size=n):
        r, o, _ = walk(hi, lo, op, int(i), float(cl[i]), int(i) + MAXHOLD)
        if r is None:
            r = cl[min(int(i) + MAXHOLD, len(cl) - 1)] / cl[i] - 1
        ctrl.append(dict(ret=r, out=o, year=df.index[int(i)].year))
c = pd.DataFrame(ctrl)

print(f"\n+{int(TARGET*100)}% / -{int(STOP*100)}%   break-even hit rate "
      f"{STOP/(TARGET+STOP)*100:.1f}%\n")
for strat, g in d.groupby("strategy"):
    print("=" * 76)
    print(f"{strat}   {len(g):,} trades, strategy hold {g.bars.mean():.1f} bars")
    print("=" * 76)
    for mode in ("within", "pure"):
        oc = g[f"{mode}_out"].value_counts(normalize=True) * 100
        t_, s_ = oc.get("target", 0), oc.get("stop", 0)
        print(f"  {mode.upper():<7} mean {g[mode].mean()*100:+7.3f}%  "
              f"median {g[mode].median()*100:+7.3f}%  (baseline {g.baseline.mean()*100:+.3f}%)")
        print(f"          target {t_:5.1f}%  stop {s_:5.1f}%  neither {oc.get('neither',0):5.1f}%"
              + (f"  -> of decided {t_/(t_+s_)*100:.1f}% hit target" if t_+s_ > 0 else ""))
    print(f"          pure held {g.pure_bars.mean():.0f} bars vs strategy {g.bars.mean():.1f}")

oc = c["out"].value_counts(normalize=True) * 100
dec = oc.get("target", 0) + oc.get("stop", 0)
print("=" * 76)
print(f"RANDOM-ENTRY CONTROL   {len(c):,} entries")
print("=" * 76)
print(f"  RANDOM  mean {c.ret.mean()*100:+7.3f}%  median {c.ret.median()*100:+7.3f}%")
print(f"          target {oc.get('target',0):5.1f}%  stop {oc.get('stop',0):5.1f}%"
      f"  -> of decided {oc.get('target',0)/dec*100:.1f}% hit target")
for strat, g in d.groupby("strategy"):
    print(f"  {strat:<16} pure edge over random "
          f"{(g['pure'].mean()-c.ret.mean())*100:+.3f}pp")
print("\n  by era (mean %):")
hdr = [(2019,2020),(2021,2022),(2023,2024),(2025,2026)]
print(f"    {'':<17}" + "".join(f"{f'{a}-{str(b)[2:]}':>10}" for a, b in hdr))
for lbl, g, col in [("RANDOM", c, "ret")] + [(s, gg, "pure") for s, gg in d.groupby("strategy")]:
    line = f"    {lbl:<17}"
    for a, b in hdr:
        m = g[(g.year >= a) & (g.year <= b)]
        line += f"{m[col].mean()*100:>+10.2f}" if len(m) > 100 else f"{'--':>10}"
    print(line)
