"""rsi2 with a +20% target and a -10% stop.

Two readings of the same instruction, because they answer different
questions and give different answers:

  WITHIN   the bracket applies inside the strategy's own trade window.
           Whichever comes first -- target, stop, or the strategy's exit.
  PURE     the bracket REPLACES the strategy's exit. Hold until +20% or
           -10%, however long that takes, capped at 250 bars so a dead
           position cannot be held forever.

FILL CONVENTIONS, and they are not symmetric. A stop is an instruction to
sell at market once the level trades, so a bar gapping through it fills at
the OPEN -- worse than the level. A target is a resting limit order, so a
bar gapping through it fills at the OPEN too -- but that is BETTER than
the level. So: stop fills min(open, level), target fills max(open, level).
Using the level for both would flatter the result at each end.

WHEN BOTH HIT IN ONE BAR the stop is assumed first. Intraday order is
unknowable from daily bars, and assuming the favourable one is how
backtests manufacture edge.

THE ARITHMETIC TO BEAT: at +20% against -10%, break-even needs a 33.3%
hit rate. Anything below that loses no matter how good it feels.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime.backtest import backtest_strategy

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
TARGET, STOP, MAXHOLD = 0.20, 0.10, 250


def walk(df, i, last, entry):
    """Return (ret, outcome, bars) for the bracket over bars i+1..last."""
    hi, lo, op = (df["High"].to_numpy(), df["Low"].to_numpy(), df["Open"].to_numpy())
    tgt, stp = entry * (1 + TARGET), entry * (1 - STOP)
    for k in range(i + 1, min(last, len(df) - 1) + 1):
        hit_s, hit_t = lo[k] <= stp, hi[k] >= tgt
        if hit_s:                                  # stop first when both
            return min(op[k], stp) / entry - 1, "stop", k - i
        if hit_t:
            return max(op[k], tgt) / entry - 1, "target", k - i
    return None, "neither", min(last, len(df) - 1) - i


rows = []
t0 = time.time()
for strat in ("rsi2", "rsi2_prior_high"):
    n = 0
    for tk, df in prices.items():
        if len(df) < 260:
            continue
        try:
            trades = backtest_strategy(df, strat, interval="1d")
        except Exception:
            continue
        pos = {d: i for i, d in enumerate(df.index)}
        for t in trades:
            i, j = pos.get(t.entry_date), pos.get(t.exit_date)
            if i is None or j is None or j <= i:
                continue
            e = float(t.entry_price)
            base = float(t.exit_price) / e - 1
            rw, ow, _ = walk(df, i, j, e)                    # within the window
            rp, op_, bp = walk(df, i, i + MAXHOLD, e)        # pure bracket
            rows.append(dict(strategy=strat, ticker=tk, entry=t.entry_date, bars=j - i,
                             baseline=base,
                             within=base if rw is None else rw, within_out=ow,
                             pure=(float(df["Close"].iloc[min(i + MAXHOLD, len(df) - 1)]) / e - 1)
                                  if rp is None else rp,
                             pure_out=op_, pure_bars=bp))
            n += 1
    print(f"  {strat}: {n:,} trades  [{time.time()-t0:.0f}s]", flush=True)

d = pd.DataFrame(rows)
d["year"] = pd.to_datetime(d["entry"]).dt.year
pickle.dump(d, open(f"{S}/bracket.pkl", "wb"))

print(f"\nbreak-even hit rate at +{int(TARGET*100)}%/-{int(STOP*100)}%: "
      f"{STOP/(TARGET+STOP)*100:.1f}%\n")
for strat, g in d.groupby("strategy"):
    print("=" * 78)
    print(f"{strat}   {len(g):,} trades, mean strategy hold {g.bars.mean():.1f} bars")
    print("=" * 78)
    for mode in ("within", "pure"):
        oc = g[f"{mode}_out"].value_counts(normalize=True) * 100
        hit = oc.get("target", 0)
        stopped = oc.get("stop", 0)
        decided = hit + stopped
        print(f"  {mode.upper():<7} mean {g[mode].mean()*100:+7.3f}%   "
              f"median {g[mode].median()*100:+7.3f}%   "
              f"(baseline {g.baseline.mean()*100:+.3f}%)")
        print(f"          target {hit:5.1f}%   stop {stopped:5.1f}%   "
              f"neither {oc.get('neither',0):5.1f}%", end="")
        if decided > 0:
            print(f"   -> of decided, {hit/decided*100:.1f}% hit target "
                  f"(need 33.3%)")
        else:
            print()
        if mode == "pure":
            print(f"          mean bars held {g.pure_bars.mean():.0f} "
                  f"vs strategy's {g.bars.mean():.1f}")
    print(f"  by era (pure, mean %):", end="")
    for a, b in ((2019, 2020), (2021, 2022), (2023, 2024), (2025, 2026)):
        m = g[(g.year >= a) & (g.year <= b)]
        print(f"  {a}-{str(b)[2:]} {m['pure'].mean()*100:+.2f}" if len(m) > 100 else "  --", end="")
    print()
