"""Path-dependent exit policies: trailing stops, trimming, averaging down.

Everything measured so far was binary -- the strategy's entry, the
strategy's exit. These policies all need the thing a Trade does not store:
the price PATH between the two. So each trade is replayed bar by bar.

The question each answers is the same one: a strategy's own exit leaves
money on the table in both directions -- it rides losers down and cuts
winners early ("good runs hit early exits"). Does intervening on the path
help, and what does it cost?

THE METRIC IS THE DISTRIBUTION, NOT THE MEAN. Every policy here takes its
cost out of the right tail, and a mean can look healthy while the few
trades that fund the strategy have been clipped. So the top decile of
winners is reported alongside, plus how much of the baseline's total
right-tail each policy preserves.

JUDGED ON CONSISTENCY, NOT BEST CELL. This is six policies across a
parameter sweep on one 7-year sample -- enough comparisons that something
will look good by chance. Each is scored across four sub-periods and a
policy that wins in one era is noise. Declared before running.

AVERAGING DOWN IS FLATTERED HERE, deliberately and unavoidably. Adding to
a loser doubles the capital in that position, which at portfolio level
means slots and cash unavailable for other trades. This replay is
trade-level, so it cannot charge that. Its numbers are an UPPER BOUND and
the capital multiple is reported beside them.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime.backtest import backtest_strategy

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
STRATS = ["ema_cross", "macd", "rsi_dip_recovery", "supertrend", "donchian"]


def replay(path: pd.DataFrame, entry: float, exit_px: float) -> dict:
    """One trade's outcome under each policy. `path` is entry bar -> exit bar.

    Stops fill at min(Open, stop): a stop is an instruction to sell at
    market once the level trades, so a bar gapping through it fills at the
    open, not at the level. Same convention as run_backtest.
    """
    hi, lo, op = path["High"].to_numpy(), path["Low"].to_numpy(), path["Open"].to_numpy()
    out = {"baseline": exit_px / entry - 1}

    def stop_exit(level_fn):
        run_hi = entry
        for i in range(1, len(path)):
            lvl = level_fn(run_hi)
            if lvl is not None and lo[i] <= lvl:
                return min(op[i], lvl) / entry - 1
            run_hi = max(run_hi, hi[i])
        return exit_px / entry - 1

    for x in (5, 10, 15, 20):
        out[f"fixed{x}"] = stop_exit(lambda rh, x=x: entry * (1 - x / 100))
        out[f"trail{x}"] = stop_exit(lambda rh, x=x: rh * (1 - x / 100))
    for y in (10, 20, 30):
        # stop to breakeven only AFTER the trade is up y%
        out[f"be{y}"] = stop_exit(
            lambda rh, y=y: entry if rh >= entry * (1 + y / 100) else None)
    for x in (20, 30, 50):
        # trim half at +x%, carry the rest to the strategy's own exit
        tgt = entry * (1 + x / 100)
        hit = np.argmax(hi >= tgt) if (hi >= tgt).any() else -1
        out[f"trim{x}"] = (0.5 * (x / 100) + 0.5 * (exit_px / entry - 1)
                           if hit > 0 else exit_px / entry - 1)
    for x in (10, 20):
        # average down: second unit at -x%, basis is the mean of the two.
        # Return is on DOUBLE the capital -- see the module docstring.
        add = entry * (1 - x / 100)
        hit = (lo <= add).any()
        out[f"avgdn{x}"] = (2 * exit_px / (entry + add) - 1) if hit else exit_px / entry - 1
        out[f"avgdn{x}_cap"] = 2.0 if hit else 1.0
    return out


print(f"backtesting {len(STRATS)} strategies over {len(prices)} tickers...", flush=True)
t0 = time.time()
rows = []
for strat in STRATS:
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
            r = replay(df.iloc[i:j + 1], float(t.entry_price), float(t.exit_price))
            r.update(strategy=strat, ticker=tk, entry=t.entry_date, bars=j - i)
            rows.append(r)
            n += 1
    print(f"  {strat}: {n:,} trades  [{time.time()-t0:.0f}s]", flush=True)

d = pd.DataFrame(rows)
d["year"] = pd.to_datetime(d["entry"]).dt.year
pickle.dump(d, open(f"{S}/exitpol.pkl", "wb"))
POL = [c for c in d.columns if c not in
       ("strategy", "ticker", "entry", "bars", "year") and not c.endswith("_cap")]

print(f"\n{len(d):,} trades, mean hold {d['bars'].mean():.0f} bars\n")
print("=" * 94)
print("ALL TRADES POOLED -- mean, and what each policy does to the right tail")
print("=" * 94)
base_tail = d.loc[d.baseline > d.baseline.quantile(0.9), "baseline"].sum()
print(f"{'policy':<12}{'mean %':>9}{'median %':>10}{'worst %':>9}{'p90 %':>8}"
      f"{'tail kept':>11}{'cut early':>11}")
for p in POL:
    v = d[p]
    tail = d.loc[d.baseline > d.baseline.quantile(0.9), p].sum()
    cut = (v < d.baseline - 1e-9).mean() * 100
    print(f"{p:<12}{v.mean()*100:>+9.3f}{v.median()*100:>+10.3f}{v.min()*100:>+9.1f}"
          f"{v.quantile(0.9)*100:>+8.2f}{tail/base_tail*100:>10.1f}%{cut:>10.1f}%")

print()
print("=" * 94)
print("BY SUB-PERIOD -- mean % per trade; a policy that wins in one era is noise")
print("=" * 94)
eras = {"2019-20": (2019, 2020), "2021-22": (2021, 2022),
        "2023-24": (2023, 2024), "2025-26": (2025, 2026)}
print(f"{'policy':<12}" + "".join(f"{k:>11}" for k in eras) + f"{'beats base':>12}")
for p in POL:
    cells, wins = [], 0
    for k, (a, b) in eras.items():
        m = d[(d.year >= a) & (d.year <= b)]
        if len(m) < 200:
            cells.append(np.nan); continue
        cells.append(m[p].mean() * 100)
        wins += int(m[p].mean() > m["baseline"].mean())
    print(f"{p:<12}" + "".join(f"{c:>+11.3f}" if not np.isnan(c) else f"{'--':>11}"
                               for c in cells) + f"{wins:>9}/4")

print()
print("=" * 94)
print("AVERAGING DOWN -- capital actually deployed")
print("=" * 94)
for x in (10, 20):
    cap = d[f"avgdn{x}_cap"].mean()
    added = (d[f"avgdn{x}_cap"] > 1).mean() * 100
    print(f"  avgdn{x}: added to {added:.1f}% of trades, mean capital {cap:.2f}x, "
          f"mean return {d[f'avgdn{x}'].mean()*100:+.3f}% vs baseline "
          f"{d['baseline'].mean()*100:+.3f}%")
    print(f"           per unit of capital deployed: "
          f"{d[f'avgdn{x}'].mean()/cap*100:+.3f}%")
