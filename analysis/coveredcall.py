"""Covered calls written INTO a drawdown: does the premium beat the bounce?

The idea under test: a position is down, you do not want to sell, so you
write a call against it to blunt the loss. The tension is specific -- in a
drawdown you are waiting for a recovery, and a covered call sells exactly
that recovery. The premium has to beat the bounce it gives up.

IV IS MODELLED, NOT OBSERVED, and that is the main limitation. No
historical options data is available on these plans, so the premium is
Black-Scholes priced off each name's own TRAILING 20-day realised vol,
scaled by 1.15 for the usual implied-over-realised premium. Using
trailing realised vol rather than a flat number matters here: vol RISES in
drawdowns, so the premium rises exactly when the call is being written,
which is the real behaviour and is favourable to the strategy. The 1.15
multiplier is a convention, not a measurement -- results scale with it.

No early assignment, held to expiry, one contract per 100 shares, and the
stock is assumed held throughout either way. So this isolates the overlay.
"""
import os, pickle, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

S = os.environ.get("SMC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
prices = pickle.load(open(f"{S}/universe_1d.pkl", "rb"))
HOLD, RF, IV_MULT = 21, 0.04, 1.15          # 21 bars ~ 30 calendar days
STRIKES = (0.03, 0.05, 0.10)
_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def bs_call_frac(moneyness, T, sig):
    """Call price as a FRACTION of spot, strike = spot*(1+moneyness)."""
    K = 1.0 + moneyness
    sig = np.clip(sig, 0.05, 2.0)
    d1 = (np.log(1.0 / K) + (RF + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return ncdf(d1) - K * math.exp(-RF * T) * ncdf(d2)


t0 = time.time()
frames = []
for tk, df in prices.items():
    if len(df) < 200:
        continue
    c = df["Close"]
    rv = c.pct_change().rolling(20).std() * math.sqrt(252)
    f = pd.DataFrame({
        "dd": (c / c.rolling(60).max() - 1) * 100,
        "iv": rv * IV_MULT,
        "fwd": (c.shift(-HOLD) / c - 1) * 100,
    }, index=df.index)
    f["ticker"] = tk
    frames.append(f.reset_index().rename(columns={"index": "date", "Date": "date"}))
d = pd.concat(frames, ignore_index=True).dropna(subset=["dd", "iv", "fwd"])
print(f"{len(d):,} ticker-bars, {time.time()-t0:.0f}s", flush=True)

T = HOLD * 1.4 / 365
for k in STRIKES:
    prem = bs_call_frac(k, T, d["iv"].to_numpy()) * 100     # % of spot
    d[f"prem{int(k*100)}"] = prem
    d[f"cc{int(k*100)}"] = np.minimum(d["fwd"].to_numpy(), k * 100) + prem

buckets = [("no DD   (> -5%)", d.dd > -5),
           ("mild   -5..-15%", (d.dd <= -5) & (d.dd > -15)),
           ("deep  -15..-30%", (d.dd <= -15) & (d.dd > -30)),
           ("severe    < -30%", d.dd <= -30)]
print(f"\n{'drawdown state':<18}{'n':>9}{'hold mean':>11}{'hold p05':>10}"
      f"{'  |':>3}{'strike':>7}{'prem%':>7}{'CC mean':>9}{'CC p05':>9}{'called%':>9}{'delta':>8}")
for name, mask in buckets:
    g = d[mask]
    if len(g) < 500:
        continue
    print(f"{name:<18}{len(g):>9,}{g.fwd.mean():>+11.2f}{g.fwd.quantile(.05):>+10.2f}", end="")
    first = True
    for k in STRIKES:
        kk = int(k * 100)
        cc = g[f"cc{kk}"]
        called = (g.fwd > k * 100).mean() * 100
        if not first:
            print(f"{'':<18}{'':>9}{'':>11}{'':>10}", end="")
        print(f"{'  |':>3}{f'+{kk}%':>7}{g[f'prem{kk}'].mean():>7.2f}"
              f"{cc.mean():>+9.2f}{cc.quantile(.05):>+9.2f}{called:>8.1f}%"
              f"{cc.mean()-g.fwd.mean():>+8.2f}")
        first = False
    print()

print("the question the policy is FOR -- does it blunt the bad outcomes?")
print(f"{'drawdown state':<18}{'strike':>8}{'hold p10':>10}{'CC p10':>9}"
      f"{'hold p25':>10}{'CC p25':>9}{'hold<0%':>9}{'CC<0%':>8}")
for name, mask in buckets:
    g = d[mask]
    if len(g) < 500:
        continue
    for k in (0.05,):
        kk = int(k * 100)
        cc = g[f"cc{kk}"]
        print(f"{name:<18}{f'+{kk}%':>8}{g.fwd.quantile(.10):>+10.2f}{cc.quantile(.10):>+9.2f}"
              f"{g.fwd.quantile(.25):>+10.2f}{cc.quantile(.25):>+9.2f}"
              f"{(g.fwd<0).mean()*100:>8.1f}%{(cc<0).mean()*100:>7.1f}%")
