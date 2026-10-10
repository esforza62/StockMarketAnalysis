"""The questions the 24-date grade history could not answer.

Same metric as smc_regime.grade_metrics -- per-date cross-sectional rank
correlation between score and forward return -- but on the reconstructed
price-only components, so roughly 1,900 dates instead of 24.

t_adj divides the raw t by sqrt(horizon bars): forward windows on
consecutive dates overlap, so adjacent per-date ICs are not independent and
the raw statistic is inflated. Sub-periods are reported because a number
averaged over seven years can hide a signal that died.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime import price_score as ps

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
HZ = {"2w": 10, "1m": 21}
COMP = list(ps.COMPONENT_MAX)

cache = f"{S}/pscore_panel.pkl"
if os.path.exists(cache):
    panel = pickle.load(open(cache, "rb"))
    print(f"loaded cached panel: {len(panel):,} rows", flush=True)
else:
    prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
    print(f"reconstructing {len(prices)} tickers (several minutes)...", flush=True)
    t0 = time.time()
    panel = ps.reconstruct(prices, progress=True)
    print(f"  done in {time.time()-t0:.0f}s, {len(panel):,} rows", flush=True)
    panel["date"] = pd.to_datetime(panel["date"])
    fwd = []
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        for name, n in HZ.items():
            g[f"fwd_{name}"] = (g["close"].shift(-n) / g["close"] - 1) * 100
        fwd.append(g)
    panel = pd.concat(fwd, ignore_index=True)
    pickle.dump(panel, open(cache, "wb"))
    print(f"  cached -> {cache}", flush=True)

panel["date"] = pd.to_datetime(panel["date"])
print(f"panel: {len(panel):,} rows, {panel['date'].nunique():,} dates, "
      f"{panel['ticker'].nunique()} tickers, "
      f"{panel['date'].min().date()} -> {panel['date'].max().date()}\n", flush=True)


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b))


def ic(sub, feat, hz, min_rows=30):
    col = f"fwd_{hz}"
    d = sub[[feat, col, "date"]].dropna()
    if d.empty:
        return None
    s = d.groupby("date").apply(lambda g: spearman(g[feat], g[col]) if len(g) >= min_rows else np.nan).dropna()
    if len(s) < 20:
        return None
    v = s.to_numpy(); m, sd = v.mean(), v.std(ddof=1)
    t = m / (sd / np.sqrt(len(v)))
    return dict(ic=m, sd=sd, hit=(v > 0).mean()*100, n=len(v),
                t=t, t_adj=t/np.sqrt(HZ[hz]))


FEATS = COMP + ["price_score"]
print("=" * 86)
print("FULL PANEL -- per-component IC  (t_adj corrects for overlapping windows)")
print("=" * 86)
print(f"{'feature':<18}{'hz':<5}{'IC':>9}{'sd':>8}{'hit':>6}{'dates':>7}{'t_raw':>8}{'t_adj':>8}")
rows = []
for f in FEATS:
    for hz in HZ:
        r = ic(panel, f, hz)
        if not r: continue
        print(f"{f:<18}{hz:<5}{r['ic']:>+9.4f}{r['sd']:>8.4f}{r['hit']:>5.0f}%"
              f"{r['n']:>7}{r['t']:>+8.1f}{r['t_adj']:>+8.1f}", flush=True)
        rows.append(dict(scope="full", feature=f, horizon=hz, **r))

print()
print("=" * 86)
print("SUB-PERIODS -- 2w, does any of it hold up out of its own era")
print("=" * 86)
periods = {"2019-2020": ("2019-01-01", "2020-12-31"),
           "2021-2022": ("2021-01-01", "2022-12-31"),
           "2023-2024": ("2023-01-01", "2024-12-31"),
           "2025-2026": ("2025-01-01", "2026-12-31")}
print(f"{'feature':<18}" + "".join(f"{k:>12}" for k in periods) + f"{'sign flips':>12}")
for f in FEATS:
    vals = []
    for k, (a, b) in periods.items():
        sub = panel[(panel.date >= a) & (panel.date <= b)]
        r = ic(sub, f, "2w")
        vals.append(r["ic"] if r else np.nan)
    good = [v for v in vals if not np.isnan(v)]
    flips = sum(1 for i in range(len(good)-1) if np.sign(good[i]) != np.sign(good[i+1]))
    print(f"{f:<18}" + "".join(f"{v:>+12.4f}" if not np.isnan(v) else f"{'--':>12}" for v in vals)
          + f"{flips:>12}", flush=True)
    for k, v in zip(periods, vals):
        rows.append(dict(scope=k, feature=f, horizon="2w", ic=v))

print()
print("=" * 86)
print("INCREMENTAL -- does the price-only score rank better WITHOUT a component")
print("=" * 86)
base = None
print(f"{'variant':<30}{'pts':>5}{'2w IC':>10}{'delta':>9}{'1m IC':>10}{'delta':>9}")
for drop in [[]] + [[c] for c in COMP]:
    keep = [c for c in COMP if c not in drop]
    panel["_v"] = panel[keep].sum(axis=1)
    r2, r1 = ic(panel, "_v", "2w"), ic(panel, "_v", "1m")
    if base is None:
        base = (r2["ic"], r1["ic"])
    label = "price-only (all six)" if not drop else f"drop {drop[0]}"
    pts = sum(ps.COMPONENT_MAX[c] for c in keep)
    print(f"{label:<30}{pts:>5}{r2['ic']:>+10.4f}{r2['ic']-base[0]:>+9.4f}"
          f"{r1['ic']:>+10.4f}{r1['ic']-base[1]:>+9.4f}", flush=True)
    rows.append(dict(scope="incremental", feature=label, horizon="2w", ic=r2["ic"],
                     delta=r2["ic"]-base[0]))

pd.DataFrame(rows).to_csv(f"{S}/pscore_ic_results.csv", index=False)
print(f"\nwrote {S}/pscore_ic_results.csv")
