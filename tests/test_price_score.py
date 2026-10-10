"""The price-only reconstruction, and the one property it must have.

This module exists to turn 24 capture dates into ~1,900, which is only
worth anything if a bar's score depends on nothing after that bar. A
look-ahead here would not crash or look wrong -- it would quietly inflate
every information coefficient measured downstream and make a dead scorer
look predictive. So the central check is a truncation test: scores computed
on the full series must equal scores computed on a prefix of it, for every
overlapping bar. If future bars leak in, they cannot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import price_score as ps
from smc_regime import regime as rg
from smc_regime.technicals import technical_snapshot

failures = []


def check(label, ok, extra=""):
    print(f"{label}  {'OK' if ok else 'FAILED'}{extra}")
    if not ok:
        failures.append(label)


def bars(n=700, seed=4):
    """Synthetic daily bars with enough history for a 200-MA and a weekly
    regime. Trend plus noise so regimes actually change."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    drift = np.concatenate([np.full(n // 2, 0.0012), np.full(n - n // 2, -0.0009)])
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.014, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(high, close),
                         "Low": np.minimum(low, close), "Close": close,
                         "Volume": rng.integers(5e5, 5e6, n).astype(float)}, index=idx)


df = bars()
full = ps.price_components(df)

# --- 1: the headline property -------------------------------------------
# Recompute on a prefix. Every bar present in both must score identically;
# anything that differs saw a bar that had not happened yet.
CUT = 520
pre = ps.price_components(df.iloc[:CUT])
cols = list(ps.COMPONENT_MAX) + ["price_score"]
# ignore the warm-up where a 200-MA does not exist in the short series
cmp_from = 260
a = full.iloc[cmp_from:CUT][cols].reset_index(drop=True)
b = pre.iloc[cmp_from:CUT][cols].reset_index(drop=True)
diff = (a - b).abs().max().max()
check("1. truncating the series does not change any earlier bar's score",
      diff < 1e-9, f"  (max drift {diff:.2e} over {CUT-cmp_from} bars x {len(cols)} cols)")

# --- 2: and the same holds at a second, earlier cut ----------------------
CUT2 = 400
pre2 = ps.price_components(df.iloc[:CUT2])
d2 = (full.iloc[cmp_from:CUT2][cols].reset_index(drop=True)
      - pre2.iloc[cmp_from:CUT2][cols].reset_index(drop=True)).abs().max().max()
check("2. the same holds at a second cut point", d2 < 1e-9, f"  (max drift {d2:.2e})")

# --- 3: the indicator frame reproduces the deployed snapshot exactly -----
v = ps.verify_against_snapshot(df)
check("3. indicator frame matches technicals.technical_snapshot on the last bar",
      v["ok"], f"  ({v['fields']} fields{'' if v['ok'] else ': ' + str(v['mismatches'])})")

# --- 4: the last row IS what the nightly would have scored ---------------
snap = technical_snapshot(df)
last = ps.indicator_frame(df).iloc[-1]
check("4. a spot field agrees to full float precision",
      abs(snap["rsi"] - float(last["rsi"])) < 1e-12)

# --- 5-6: streak matches the deployed helper -----------------------------
conf = rg.confirmed_regime(rg.classify_regime(df))
check("5. the vectorised streak matches regime_streak_bars on the last bar",
      int(ps._streak_series(conf).iloc[-1]) == rg.regime_streak_bars(conf),
      f"  ({int(ps._streak_series(conf).iloc[-1])} bars)")
s = ps._streak_series(conf)
key = conf["regime"].astype(str) + "|" + conf["direction"].astype(str)
restart = key != key.shift()
check("6. the streak restarts at 1 exactly when the regime pair changes",
      bool((s[restart] == 1).all()) and bool((s[~restart] > 1).all()))

# --- 7-9: the weekly rung cannot see the week a day sits inside ----------
# Asserted STRUCTURALLY, not by mutation. A mutation test here passes
# vacuously whenever the altered week happens to classify the same way,
# and an earlier version of this file did exactly that -- it let a deleted
# shift(1) through. So instead: prove where each value came from.
#
# The weekly index is week-ending-Friday, ffilled onto days. A Monday
# already reads the prior Friday's bar. The exposed bar is a FRIDAY, whose
# date matches its own week's label exactly, so that is where a missing
# shift shows up.
wk = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                               "Close": "last", "Volume": "sum"}).dropna()
wkconf = rg.confirmed_regime(rg.classify_regime(wk))
wr = ps._weekly_regime(df)

fridays = [i for i, d in enumerate(df.index) if d.weekday() == 4 and i > 320]
bad_src = []
for i in fridays:
    day = df.index[i]
    got = (wr["regime"].iloc[i], wr["direction"].iloc[i])
    if got[0] == "n/a":
        continue
    # which weekly bars could legitimately have produced this? only those
    # whose week ENDED STRICTLY BEFORE this day
    allowed = wkconf[wkconf.index < day]
    if allowed.empty:
        continue
    want = (allowed["regime"].iloc[-1], allowed["direction"].iloc[-1])
    if got != want:
        bad_src.append((day.date(), got, want))

check("7. every Friday's weekly read comes from a week that ENDED before it",
      not bad_src,
      f"  (checked {len(fridays)} Fridays"
      + (f"; first bad {bad_src[0]}" if bad_src else "") + ")")

# The invariant that actually matters, as EXACT equality against an
# independently computed expectation: every daily bar must carry the
# confirmed regime of the last week that ended strictly before it. Stated
# this way it catches both failure directions at once -- a look-ahead reads
# a week that has not closed, and shift(1)+ffill reads one a week too old
# for any day that is not a Friday. A looser "is it one of the allowed
# weeks" form passes vacuously whenever consecutive weeks share a regime,
# which is most of the time; an earlier version of this file did that and
# let the double-lag through.
bad = []
for i in range(320, len(df)):
    day = df.index[i]
    allowed = wkconf[wkconf.index < day]
    if allowed.empty or wr["regime"].iloc[i] == "n/a":
        continue
    want = (allowed["regime"].iloc[-1], allowed["direction"].iloc[-1])
    got = (wr["regime"].iloc[i], wr["direction"].iloc[i])
    if got != want:
        bad.append((day.date(), got, want))
check("8. every daily bar carries exactly the last week that CLOSED before it",
      not bad,
      f"  ({len(df)-320} bars checked"
      + (f", {len(bad)} wrong, first {bad[0]}" if bad else ", 0 wrong") + ")")

check("9. the weekly frame is aligned to the daily index",
      len(wr) == len(df) and bool((wr.index == df.index).all()))

# --- 10-15: structure -----------------------------------------------------
check("10. price_score is the sum of the six components",
      bool(((full[list(ps.COMPONENT_MAX)].sum(axis=1) - full["price_score"]).abs() < 1e-9).all()))
check("11. the scale is 85 points, not 100",
      ps.PRICE_ONLY_MAX == 85 and len(ps.COMPONENT_MAX) == 6)
check("12. no component ever exceeds its documented maximum",
      all(bool((full[c].dropna() <= m + 1e-9).all()) for c, m in ps.COMPONENT_MAX.items()))
check("13. and none is negative",
      all(bool((full[c].dropna() >= -1e-9).all()) for c in ps.COMPONENT_MAX))

# --- 12: a short series is skipped rather than scored on nothing ---------
out = ps.reconstruct({"SHORT": df.iloc[:50], "OK": df}, min_bars=260)
check("14. reconstruct skips a series too short to score",
      set(out["ticker"].unique()) == {"OK"})
check("15. reconstruct returns an empty frame rather than raising on no input",
      ps.reconstruct({}).empty)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
