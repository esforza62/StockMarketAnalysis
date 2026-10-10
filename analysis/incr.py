"""Does dropping a component IMPROVE the grade? The incremental test.

A component's standalone IC says whether it ranks on its own. It does not
say whether the grade is better with it, because a weak-but-orthogonal
component can still add and a strong-but-redundant one can be dead weight.
The only question that matters for the scorer is whether the TOTAL ranks
better without it.

Scores are recomputed from the DEPLOYED per-component points in the grade
history, so every variant is the real scorer minus real points -- no
reimplementation, no rescaling, and the baseline reproduces total_points
exactly (checked below).

LOW POWER, SAY IT TWICE. 14 usable dates at 2w and 5 at 1m, with
overlapping forward windows, so the effective independent sample is 1-2
observations. Treat a gap of less than ~0.02 IC as noise. This is a
consistency check against the standalone table, not evidence on its own.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from smc_regime import grade_history as gh, grade_report as gr

COMPS = ["trend_structure", "rsi", "macd", "volume", "streak", "alignment",
         "sector_industry", "valuation"]


def spearman(a, b):
    a, b = pd.Series(list(a)).rank(), pd.Series(list(b)).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b))


h = gh.load()
h = h if isinstance(h, pd.DataFrame) else pd.DataFrame(h)
s = gr.forward_returns(h)
for c in COMPS:
    s[f"{c}_points"] = pd.to_numeric(s[f"{c}_points"], errors="coerce")

full = s[[f"{c}_points" for c in COMPS]].sum(axis=1)
err = (full - pd.to_numeric(s["total_points"], errors="coerce")).abs().max()
print(f"baseline reproduces total_points to {err:.4f} points "
      f"({'exact enough' if err < 0.15 else 'MISMATCH -- investigate'})\n")

VARIANTS = {
    "full grade (baseline)": [],
    "drop valuation": ["valuation"],
    "drop rsi": ["rsi"],
    "drop streak": ["streak"],
    "drop sector_industry": ["sector_industry"],
    "drop volume": ["volume"],
    "drop valuation+rsi": ["valuation", "rsi"],
    "drop valuation+rsi+streak": ["valuation", "rsi", "streak"],
    "PRICE-ONLY (drop sec_ind+val)": ["sector_industry", "valuation"],
    "price-only, drop rsi too": ["sector_industry", "valuation", "rsi"],
    "trend+macd+alignment only": ["rsi", "volume", "streak", "sector_industry", "valuation"],
}

print(f"{'variant':<32}{'pts':>5}{'2w IC':>9}{'hit':>6}{'1m IC':>9}{'hit':>6}{'d2w':>8}")
base = {}
rows = []
for label, drop in VARIANTS.items():
    keep = [c for c in COMPS if c not in drop]
    score = s[[f"{c}_points" for c in keep]].sum(axis=1)
    tmp = s.assign(_v=score)
    res = {}
    for hz in ("2w", "1m"):
        col = f"fwd_{hz}"
        u = tmp[tmp[f"mature_{hz}"] & tmp[col].notna()]
        ics = u.groupby("as_of").apply(
            lambda g: spearman(g["_v"], g[col]) if len(g) >= 30 else np.nan).dropna()
        v = ics.to_numpy()
        res[hz] = (v.mean() if len(v) else np.nan,
                   (v > 0).mean() * 100 if len(v) else np.nan, len(v))
    if not base:
        base = res
    d2w = res["2w"][0] - base["2w"][0]
    # max points available for that variant
    mx = {"trend_structure": 20, "rsi": 15, "macd": 15, "volume": 15,
          "streak": 10, "alignment": 10, "sector_industry": 5, "valuation": 10}
    pts = sum(mx[c] for c in keep)
    print(f"{label:<32}{pts:>5}{res['2w'][0]:>+9.4f}{res['2w'][1]:>5.0f}%"
          f"{res['1m'][0]:>+9.4f}{res['1m'][1]:>5.0f}%{d2w:>+8.4f}")
    rows.append(dict(variant=label, points=pts, ic_2w=res["2w"][0], hit_2w=res["2w"][1],
                     ic_1m=res["1m"][0], hit_1m=res["1m"][1], delta_2w=d2w,
                     dates_2w=res["2w"][2], dates_1m=res["1m"][2]))

pd.DataFrame(rows).to_csv(f"{os.environ.get('SMC_CACHE', os.path.dirname(os.path.abspath(__file__)))}/incr_results.csv", index=False)
print(f"\ndates: {rows[0]['dates_2w']} at 2w, {rows[0]['dates_1m']} at 1m "
      f"-- effective independent sample 1-2. Gaps under ~0.02 are noise.")
