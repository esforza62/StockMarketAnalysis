"""The control PURE mode needs: does the RSI(2) entry matter at all?

Holding until +20% or -10% takes ~58 bars. Over 2019-2026, in a universe
of 415 tickers that still exist today, almost any entry held three months
did well. So the comparison that matters is not "bracket vs the strategy's
own exit" -- it is "bracket from an RSI(2) entry vs bracket from a RANDOM
entry in the same names over the same period".

If random matches it, the signal contributes nothing and the result is the
market plus survivorship.
"""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
real = pickle.load(open(f"{S}/bracket.pkl", "rb"))
TARGET, STOP, MAXHOLD = 0.20, 0.10, 250
rng = np.random.default_rng(7)


def walk(hi, lo, op, i, entry, last):
    tgt, stp = entry * (1 + TARGET), entry * (1 - STOP)
    for k in range(i + 1, min(last, len(hi) - 1) + 1):
        if lo[k] <= stp:
            return min(op[k], stp) / entry - 1, "stop"
        if hi[k] >= tgt:
            return max(op[k], tgt) / entry - 1, "target"
    return None, "neither"


# match the real entry COUNT per ticker so composition is identical
counts = real[real.strategy == "rsi2"].groupby("ticker").size().to_dict()
rows = []
for tk, n in counts.items():
    df = prices.get(tk)
    if df is None or len(df) < 300:
        continue
    hi, lo, op, cl = (df["High"].to_numpy(), df["Low"].to_numpy(),
                      df["Open"].to_numpy(), df["Close"].to_numpy())
    idx = rng.integers(0, max(1, len(df) - 2), size=n)
    for i in idx:
        r, o = walk(hi, lo, op, int(i), float(cl[i]), int(i) + MAXHOLD)
        if r is None:
            j = min(int(i) + MAXHOLD, len(cl) - 1)
            r = cl[j] / cl[i] - 1
        rows.append(dict(ticker=tk, ret=r, out=o, year=df.index[int(i)].year))

c = pd.DataFrame(rows)
print(f"RANDOM-ENTRY control: {len(c):,} entries, same tickers, same counts\n")
oc = c["out"].value_counts(normalize=True) * 100
dec = oc.get("target", 0) + oc.get("stop", 0)
print(f"  RANDOM   mean {c.ret.mean()*100:+7.3f}%   median {c.ret.median()*100:+7.3f}%")
print(f"           target {oc.get('target',0):5.1f}%   stop {oc.get('stop',0):5.1f}%"
      f"   neither {oc.get('neither',0):5.1f}%   -> of decided "
      f"{oc.get('target',0)/dec*100:.1f}% hit target")
for strat, g in real.groupby("strategy"):
    print(f"  {strat:<16} mean {g['pure'].mean()*100:+7.3f}%   "
          f"edge over random {(g['pure'].mean()-c.ret.mean())*100:+.3f}pp")
print("\n  by era (mean %):")
print(f"    {'':<16}" + "".join(f"{a}-{str(b)[2:]:>8}" for a, b in
      ((2019,2020),(2021,2022),(2023,2024),(2025,2026))))
for lbl, g, col in [("RANDOM", c, "ret")] + [(s, gg, "pure") for s, gg in real.groupby("strategy")]:
    line = f"    {lbl:<16}"
    for a, b in ((2019,2020),(2021,2022),(2023,2024),(2025,2026)):
        m = g[(g.year >= a) & (g.year <= b)]
        line += f"{m[col].mean()*100:>+10.2f}" if len(m) > 100 else f"{'--':>10}"
    print(line)
