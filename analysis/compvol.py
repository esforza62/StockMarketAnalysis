"""Two questions: which grade component earns its weight, and does volume
read better as a ladder of nested windows than as one 5-vs-50 ratio.

PART A uses the deployed per-component scores out of the grade history, so
it measures what the scorer actually did rather than a reimplementation.
Only 24 capture dates exist, so it is low-powered by construction and the
dispersion matters more than the point estimate.

PART B is price-only and runs over the full 2019-2026 panel. The current
volume component is one ratio -- last 5 bars' average against the last 50.
The proposal is a cascade: 30 against 50, 10 against 30, 3 against 10, so
that "volume is building" is a shape across scales rather than a single
number. Each rung is tested alone, plus the strict case where all three
point the same way.

Volume is also tested signed by price direction, because the deployed
component is not a volume reading at all -- it is volume AND direction
together ("advance on heavy volume -- conviction"), and an unsigned ratio
would be measuring a different thing from what the grade scores.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime import grade_history as gh, grade_report as gr

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))


def spearman(a, b):
    a, b = pd.Series(list(a)).rank(), pd.Series(list(b)).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b))


def ic_table(panel, feats, horizons, min_rows=30, label=""):
    print(f"\n{label}")
    print(f"{'feature':<22}{'hz':<5}{'IC mean':>9}{'sd':>8}{'hit':>6}{'dates':>7}{'t_raw':>7}{'t_adj':>7}")
    out = []
    for f in feats:
        for hz, bars in horizons.items():
            sub = panel[[f, f"fwd_{hz}", "date"]].dropna()
            if sub.empty:
                continue
            ics = sub.groupby("date").apply(
                lambda g: spearman(g[f], g[f"fwd_{hz}"]) if len(g) >= min_rows else np.nan).dropna()
            if len(ics) < 5:
                continue
            v = ics.to_numpy(); m, sd = v.mean(), v.std(ddof=1)
            t = m / (sd / np.sqrt(len(v)))
            # forward windows overlap by `bars`, so adjacent dates share
            # return data and the raw t is inflated by about sqrt(bars)
            t_adj = t / np.sqrt(max(bars, 1))
            print(f"{f:<22}{hz:<5}{m:>+9.4f}{sd:>8.4f}{(v>0).mean()*100:>5.0f}%"
                  f"{len(v):>7}{t:>+7.1f}{t_adj:>+7.1f}", flush=True)
            out.append(dict(feature=f, horizon=hz, ic_mean=m, ic_sd=sd,
                            hit=(v > 0).mean()*100, dates=len(v), t_raw=t, t_adj=t_adj))
    return out


# ============================ PART A ====================================
print("=" * 78)
print("PART A -- per-component IC, from the DEPLOYED grade history (24 dates)")
print("=" * 78)
h = gh.load()
h = h if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
scored = gr.forward_returns(h)
comp = [c for c in h.columns if c.endswith("_points")]
pa = scored.rename(columns={"as_of": "date"})
for hz in ("1w", "2w", "1m"):
    pa[f"fwd_{hz}"] = np.where(pa[f"mature_{hz}"], pa[f"fwd_{hz}"], np.nan)
rows_a = ic_table(pa, comp, {"2w": 10, "1m": 21},
                  label="component (deployed points) vs forward return")

# ============================ PART B ====================================
print()
print("=" * 78)
print("PART B -- volume as a ladder, full 2019-2026 panel")
print("=" * 78)
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))


def vol_feats(df, weekly=False):
    v, c = df["Volume"], df["Close"]
    f = pd.DataFrame(index=df.index)
    m = {n: v.rolling(n).mean() for n in (3, 5, 10, 30, 50)}
    f["vol_cur_5v50"] = m[5] / m[50]           # what the grade uses today
    f["vol_30v50"] = m[30] / m[50]
    f["vol_10v30"] = m[10] / m[30]
    f["vol_3v10"] = m[3] / m[10]
    # the cascade as one number: how many rungs are building
    rungs = ((f.vol_30v50 > 1).astype(int) + (f.vol_10v30 > 1).astype(int)
             + (f.vol_3v10 > 1).astype(int))
    f["vol_rungs_up"] = rungs
    f["vol_cascade"] = np.where(rungs == 3, 1, np.where(rungs == 0, -1, 0))
    # signed by direction -- the deployed component scores volume AND
    # direction together, so unsigned volume measures something else
    win = 5 if not weekly else 3
    d = np.sign(c.pct_change(win))
    for k in ("vol_cur_5v50", "vol_30v50", "vol_10v30", "vol_3v10", "vol_rungs_up"):
        f[f"{k}_signed"] = f[k] * d
    return f


FEATS = ["vol_cur_5v50", "vol_30v50", "vol_10v30", "vol_3v10", "vol_rungs_up",
         "vol_cascade", "vol_cur_5v50_signed", "vol_30v50_signed",
         "vol_10v30_signed", "vol_3v10_signed", "vol_rungs_up_signed"]

for tf, weekly in (("1d", False), ("1wk", True)):
    t0 = time.time()
    rows = []
    for tk, df in prices.items():
        if len(df) < 320:
            continue
        d = df
        if weekly:
            d = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                          "Close": "last", "Volume": "sum"}).dropna()
            if len(d) < 70:
                continue
        f = vol_feats(d, weekly)
        c = d["Close"]
        # horizons in BARS of that timeframe: ~2w and ~1m either way
        hz = {"2w": 10, "1m": 21} if not weekly else {"2w": 2, "1m": 4}
        for name, n in hz.items():
            f[f"fwd_{name}"] = (c.shift(-n) / c - 1) * 100
        f["ticker"] = tk
        rows.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
    panel = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    hz = {"2w": 10, "1m": 21} if not weekly else {"2w": 2, "1m": 4}
    print(f"\n[{tf}] {len(rows)} tickers, {len(panel):,} rows, "
          f"{panel['date'].nunique():,} dates  ({time.time()-t0:.0f}s)")
    r = ic_table(panel, FEATS, hz, label=f"volume ladder on {tf} bars")
    for x in r:
        x["timeframe"] = tf
    rows_a += r

pd.DataFrame(rows_a).to_csv(f"{S}/compvol_results.csv", index=False)
print(f"\nwrote {S}/compvol_results.csv")
