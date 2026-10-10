"""Does a richer trend-structure read rank better than the current one?

The grade scores trend structure in 4 position states against the 50/200
MA, for 20 of its 100 points. The proposal is to add an EMA stack, swing
structure (HH/HL vs LH/LL), and agreement across timeframes.

Measured, not assumed. These features are PRICE-ONLY, so unlike the grade
itself they can be tested over the full 2019-2026 history across 415
tickers rather than the 24 capture dates the grade history holds -- which
is both far more power and immune to the overfitting trap that a 24-date
fit would walk into.

The number is a cross-sectional information coefficient: on each date, rank
every ticker by the feature and correlate against its forward return. One
IC per date, then the mean, dispersion and hit rate -- the same metric
smc_regime.grade_metrics tracks for the grade.
"""
import os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smc_regime import structure as st

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
HORIZONS = {"2w": 10, "1m": 21}          # trading bars
SWING_LEN = 5


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    f = pd.DataFrame(index=df.index)

    # --- baseline: what the grade does today, as a single ordinal --------
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    pos = np.where((c > ma50) & (c > ma200), 3,
          np.where(c > ma200, 2,
          np.where(c > ma50, 1, 0)))
    f["base_ma"] = pos + np.where(ma50 > ma200, 0.5, 0)

    # --- EMA stack: how much of 8>21>50>200 holds, signed ----------------
    e8, e21, e50, e200 = (ema(c, n) for n in (8, 21, 50, 200))
    up = (e8 > e21).astype(int) + (e21 > e50).astype(int) + (e50 > e200).astype(int)
    dn = (e8 < e21).astype(int) + (e21 < e50).astype(int) + (e50 < e200).astype(int)
    f["ema_stack"] = up - dn                                   # -3..+3
    f["ema_stack_px"] = f["ema_stack"] + np.sign(c - e21)      # + where price sits

    # --- swing structure: consecutive HH/HL vs LL/LH ---------------------
    # structure.swing_points stamps a pivot at the bar it became KNOWABLE,
    # not the bar it describes, so this carries no look-ahead.
    sp = st.swing_points(df, length=SWING_LEN)
    hp = sp["high_price"].replace(0, np.nan).ffill()
    lp = sp["low_price"].replace(0, np.nan).ffill()
    # a new pivot is confirmed wherever the ffilled level changes
    newh = hp.ne(hp.shift())
    newl = lp.ne(lp.shift())
    hh = (newh & (hp > hp.shift())).astype(int) - (newh & (hp < hp.shift())).astype(int)
    hl = (newl & (lp > lp.shift())).astype(int) - (newl & (lp < lp.shift())).astype(int)
    # running structure score over the last few confirmed pivots
    f["swing_struct"] = (hh.rolling(60, min_periods=1).sum()
                         + hl.rolling(60, min_periods=1).sum())
    # strict: consecutive run of same-signed pivot moves
    sig = np.sign(hh + hl).replace(0, np.nan).ffill().fillna(0)
    grp = (sig != sig.shift()).cumsum()
    f["swing_run"] = sig * sig.groupby(grp).cumcount().add(1).clip(upper=6)

    # --- weekly rung ------------------------------------------------------
    wk = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum"}).dropna()
    if len(wk) > 60:
        wc = wk["Close"]
        we8, we21, we50 = (ema(wc, n) for n in (8, 21, 50))
        wstack = ((we8 > we21).astype(int) + (we21 > we50).astype(int)
                  - (we8 < we21).astype(int) - (we21 < we50).astype(int))
        # as-of the last CLOSED weekly bar: shift one week, then ffill onto days
        f["wk_stack"] = wstack.shift(1).reindex(df.index, method="ffill")
    else:
        f["wk_stack"] = np.nan

    # --- the user's claim: agreement across rungs ------------------------
    d_dir = np.sign(f["ema_stack"])
    w_dir = np.sign(f["wk_stack"])
    s_dir = np.sign(f["swing_struct"])
    f["rungs_agree"] = (d_dir + w_dir + s_dir)                  # -3..+3
    # strict: only when ALL THREE point the same way, else 0
    allsame = (d_dir == w_dir) & (w_dir == s_dir)
    f["rungs_unanimous"] = np.where(allsame, d_dir * 3, 0)
    return f


FEATS = ["base_ma", "ema_stack", "ema_stack_px", "swing_struct", "swing_run",
         "wk_stack", "rungs_agree", "rungs_unanimous"]

print(f"computing features for {len(prices)} tickers...", flush=True)
t0 = time.time()
rows = []
for tk, df in prices.items():
    if len(df) < 260:
        continue
    try:
        f = features(df)
    except Exception as e:
        continue
    c = df["Close"]
    for name, n in HORIZONS.items():
        f[f"fwd_{name}"] = (c.shift(-n) / c - 1) * 100
    f["ticker"] = tk
    rows.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
print(f"  {len(rows)} tickers in {time.time()-t0:.0f}s", flush=True)

panel = pd.concat(rows, ignore_index=True)
panel = panel.dropna(subset=["date"])
print(f"  panel {len(panel):,} rows, {panel['date'].nunique():,} dates\n", flush=True)


def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b))


print(f"{'feature':<18}{'horizon':<8}{'IC mean':>9}{'sd':>8}{'hit':>7}{'dates':>8}{'t-stat':>9}")
out = []
for feat in FEATS:
    for hz in HORIZONS:
        col = f"fwd_{hz}"
        sub = panel[[feat, col, "date"]].dropna()
        if sub.empty:
            continue
        ics = sub.groupby("date").apply(
            lambda g: spearman(g[feat], g[col]) if len(g) >= 30 else np.nan)
        ics = ics.dropna().to_numpy()
        if len(ics) < 10:
            continue
        m, sd = ics.mean(), ics.std(ddof=1)
        t = m / (sd / np.sqrt(len(ics)))
        print(f"{feat:<18}{hz:<8}{m:>+9.4f}{sd:>8.4f}{(ics>0).mean()*100:>6.0f}%"
              f"{len(ics):>8}{t:>+9.1f}", flush=True)
        out.append(dict(feature=feat, horizon=hz, ic_mean=m, ic_sd=sd,
                        hit=(ics > 0).mean() * 100, dates=len(ics), t=t))
pd.DataFrame(out).to_csv(f"{S}/trendfeat_results.csv", index=False)
print(f"\nwrote {S}/trendfeat_results.csv")
