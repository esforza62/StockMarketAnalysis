"""Walk-forward the PORTFOLIO, not just one split.

The OOS verdict rested on a single boundary (2024-01-01) whose test window
was a strong persistent uptrend -- structurally hostile to mean reversion.
This repeats the identical select/evaluate protocol across sequential
folds, so "the edge does not survive costs" can be told apart from "the
edge is regime-dependent".

Expanding-window train, one-year test. Selection inside each fold sees
ONLY that fold's in-sample window and uses the same pre-registered rule as
before: highest CAGR among configurations whose drawdown is no worse than
the benchmark's, falling back to highest CAGR outright.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime.backtest import backtest_strategy
from smc_regime.portfolio import simulate_seeds
from smc_regime.portfolio_cli import split_trades

# Where the cached universe/trade pickles live. Defaults beside this file so
# a fresh checkout works; override with SMC_CACHE to reuse an existing cache.
S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
TZ = prices["SPY"].index.tz
STRATS = ("rsi2", "rsi2_prior_high")
SLOTS = (10, 20, 40)
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
FOLDS = [f"{y}-01-01" for y in (2022, 2023, 2024, 2025, 2026)]


def cagr_dd(close):
    yrs = (close.index[-1] - close.index[0]).days / 365.25
    return (((close.iloc[-1] / close.iloc[0]) ** (1 / yrs) - 1) * 100,
            float(((close / close.cummax()) - 1).min() * 100))


print(f"backtesting {len(prices)} tickers x {len(STRATS)} strategies (once, full history)...")
t0 = time.time()
ALL = {}
for s in STRATS:
    ALL[s] = {}
    for tk, df in prices.items():
        if len(df) < 60:
            continue
        try:
            ALL[s][tk] = backtest_strategy(df, s, interval="1d")
        except Exception:
            continue
    print(f"  {s}: {sum(map(len, ALL[s].values()))} trades over {len(ALL[s])} tickers")
print(f"backtest took {time.time()-t0:.0f}s\n")
pickle.dump(ALL, open(f"{S}/wf_trades.pkl", "wb"))

spy = prices["SPY"]["Close"]
rows = []
for i, b in enumerate(FOLDS):
    split = pd.Timestamp(b, tz=TZ)
    oos_end = pd.Timestamp(FOLDS[i + 1], tz=TZ) if i + 1 < len(FOLDS) else spy.index[-1] + pd.Timedelta(days=1)
    t0 = time.time()

    p_is = {t: d[d.index < split] for t, d in prices.items()}
    p_oos = {t: d[(d.index >= split) & (d.index < oos_end)] for t, d in prices.items()}
    s_is = spy[spy.index < split]
    s_oos = spy[(spy.index >= split) & (spy.index < oos_end)]
    if len(s_oos) < 60:
        print(f"fold {b}: OOS too short ({len(s_oos)} bars), skipped"); continue
    is_c, is_dd = cagr_dd(s_is)
    oos_c, oos_dd = cagr_dd(s_oos)
    # trend strength of the TEST window: share of days SPY closed above its
    # own 200-day average. The hypothesis under test is that mean reversion
    # fares worse the more persistent the trend.
    sma = spy.rolling(200).mean()
    above = float((s_oos > sma.reindex(s_oos.index)).mean() * 100)

    scored, oos_tr = [], {}
    for s in STRATS:
        tr_is, tr_oos, dropped = {}, {}, 0
        for tk, trs in ALL[s].items():
            a, _, d1 = split_trades(trs, split)
            b2, _, d2 = split_trades([t for t in trs if t.entry_date >= split], oos_end)
            tr_is[tk], tr_oos[tk], dropped = a, b2, dropped + d1 + d2
        oos_tr[s] = tr_oos
        for n in SLOTS:
            sw = simulate_seeds(tr_is, p_is, seeds=SEEDS, max_positions=n, slippage_pct=0.05)
            c, d = sw.spread("cagr_pct"), sw.spread("max_drawdown_pct")
            scored.append((c["mean"], d["mean"], s, n))

    clearing = [x for x in scored if x[1] >= is_dd]
    pick = max(clearing or scored, key=lambda x: x[0])
    _, pick_dd, ps, pn = pick
    basis = "constraint" if clearing else "fallback"

    out = {"fold": b, "oos_end": str(oos_end.date()), "pick": f"{ps}@{pn}", "basis": basis,
           "spy_is_cagr": is_c, "spy_is_dd": is_dd, "spy_oos_cagr": oos_c, "spy_oos_dd": oos_dd,
           "pct_above_200sma": above, "is_cagr": pick[0], "is_dd": pick_dd}
    for bp in (0.0, 0.05, 0.10):
        sw = simulate_seeds(oos_tr[ps], p_oos, seeds=SEEDS, max_positions=pn, slippage_pct=bp)
        c, d = sw.spread("cagr_pct"), sw.spread("max_drawdown_pct")
        tag = int(bp * 100)
        out[f"oos{tag}"] = c["mean"]; out[f"oos{tag}_sd"] = c["sd"]
        out[f"oos{tag}_min"] = c["min"]; out[f"oos{tag}_max"] = c["max"]
        out[f"dd{tag}"] = d["mean"]
        out[f"rej{tag}"] = float(np.mean([r.rejection_rate for r in sw.runs]) * 100)
    rows.append(out)
    print(f"fold {b} -> {out['oos_end']}  pick {ps}@{pn} ({basis})  "
          f"SPY OOS {oos_c:+.2f}%  strat 5bp {out['oos5']:+.2f}%  "
          f"[{time.time()-t0:.0f}s]", flush=True)

pd.DataFrame(rows).to_csv(f"{S}/wf_results.csv", index=False)
print(f"\nwrote {S}/wf_results.csv")
