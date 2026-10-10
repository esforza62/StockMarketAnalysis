"""Does the grade predict whether the CURRENT REGIME CONTINUES?

Everything measured so far scored the grade against forward returns. This
asks the different question a regime tool is actually for: at bar t, is the
confirmed (regime, direction) still the same N bars later?

Three things decide whether an answer here means anything.

THE BASELINE IS THE WHOLE GAME. Regimes persist on their own. If 70% of
bars keep their label for ten bars, a grade that picks 72% has told you
almost nothing, and quoting 72% without the 70% would be misleading.

MEASURE WITHIN REGIME TYPE. Trending bars probably persist more than
choppy ones, and the grade probably scores higher in trending. Pooled, the
grade would then look predictive while only re-reading the regime label the
caller already has. The within-regime AUC is the honest number.

AUC, NOT IC. This is a binary outcome, so the rank statistic is the
Mann-Whitney one. 0.50 is nothing. Computed from ranks directly -- sklearn
is not a dependency here.
"""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
panel = pickle.load(open(f"{S}/pscore_panel.pkl", "rb"))
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["ticker", "date"])
COMP = ["trend_structure", "rsi", "macd", "volume", "streak", "alignment"]
HZ = {"2w": 10, "1m": 21}

key = panel["regime"].astype(str) + "|" + panel["direction"].astype(str)
panel["key"] = key
for name, n in HZ.items():
    # float, not bool: the last n bars of each ticker have no future label
    # and must be NaN, which pandas refuses to put in a bool column.
    fut_key = panel.groupby("ticker")["key"].shift(-n)
    panel[f"same_{name}"] = (fut_key == panel["key"]).astype(float).where(fut_key.notna())
    fut_dir = panel.groupby("ticker")["direction"].shift(-n)
    panel[f"dir_{name}"] = ((fut_dir.astype(str) == panel["direction"].astype(str))
                            .astype(float).where(fut_dir.notna()))


def auc(score, label):
    """Mann-Whitney AUC. 0.50 = no separation."""
    m = pd.notna(score) & pd.notna(label)
    s, y = pd.Series(score)[m], pd.Series(label)[m].astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0 or len(s) < 50:
        return np.nan, npos + nneg
    r = s.rank()
    return float((r[y.to_numpy()].sum() - npos * (npos + 1) / 2) / (npos * nneg)), npos + nneg


print("=" * 80)
print("BASELINE -- how often does a regime label survive on its own?")
print("=" * 80)
for name in HZ:
    v = panel[f"same_{name}"].dropna()
    d = panel[f"dir_{name}"].dropna()
    print(f"  {name}: same (regime,direction) {v.mean()*100:.1f}%   "
          f"same direction only {d.mean()*100:.1f}%   n={len(v):,}")

print()
print("by regime type, 2w, same (regime,direction):")
for r, g in panel.groupby("regime"):
    v = g["same_2w"].dropna()
    if len(v) > 500:
        print(f"  {r:<12}{v.mean()*100:>6.1f}%   n={len(v):>8,}")

print()
print("=" * 80)
print("POOLED AUC -- and why it cannot be taken at face value")
print("=" * 80)
print(f"{'feature':<18}{'2w AUC':>9}{'1m AUC':>9}{'n':>12}")
for f in ["price_score"] + COMP:
    a2, n2 = auc(panel[f], panel["same_2w"])
    a1, _ = auc(panel[f], panel["same_1m"])
    print(f"{f:<18}{a2:>9.4f}{a1:>9.4f}{n2:>12,}")

print()
print("=" * 80)
print("WITHIN REGIME TYPE -- the honest number")
print("=" * 80)
regimes = [r for r, g in panel.groupby("regime") if len(g) > 20000]
print(f"{'feature':<18}" + "".join(f"{r[:10]:>12}" for r in regimes) + f"{'spread':>9}")
for f in ["price_score"] + COMP:
    vals = []
    for r in regimes:
        a, _ = auc(panel.loc[panel.regime == r, f], panel.loc[panel.regime == r, "same_2w"])
        vals.append(a)
    good = [v for v in vals if not np.isnan(v)]
    print(f"{f:<18}" + "".join(f"{v:>12.4f}" if not np.isnan(v) else f"{'--':>12}" for v in vals)
          + f"{(max(good)-min(good)) if good else float('nan'):>9.4f}")

print()
print("=" * 80)
print("PERSISTENCE BY SCORE QUARTILE, within regime (2w)")
print("=" * 80)
for r in regimes:
    sub = panel[panel.regime == r].copy()
    sub["q"] = pd.qcut(sub["price_score"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                       duplicates="drop")
    t = sub.groupby("q", observed=True)["same_2w"].agg(["mean", "size"])
    base = sub["same_2w"].mean()
    line = "  ".join(f"{q} {row['mean']*100:.1f}%" for q, row in t.iterrows())
    top = t["mean"].iloc[-1] * 100 if len(t) else float("nan")
    bot = t["mean"].iloc[0] * 100 if len(t) else float("nan")
    print(f"{r:<12} base {base*100:.1f}%   {line}   Q4-Q1 {top-bot:+.1f}pp")

print()
print("=" * 80)
print("DOES A LONGER STREAK MEAN MORE PERSISTENCE? (2w, within regime)")
print("=" * 80)
for r in regimes:
    sub = panel[panel.regime == r]
    a, n = auc(sub["streak"], sub["same_2w"])
    print(f"  {r:<12} streak AUC {a:.4f}   n={n:,}")
