"""Entry mechanics for smc_regime.strategies.engulfer_rsi_confirmed.

Synthetic bars -- no Tiingo key needed. Every fixture is the same shape: a
long decline (so RSI is weak), a down bar, a bullish engulfer, then a rally
whose steepness decides whether the RSI confirmation arrives cheaply or only
after price has already run.
"""
import numpy as np
import pandas as pd

from smc_regime import indicators as ind
from smc_regime.backtest import run_backtest
from smc_regime.strategies import _outside_bar_signals, engulfer_rsi_confirmed


def build(rally_step, rally_bars=25):
    opens, highs, lows, closes = [], [], [], []
    px = 100.0
    for i in range(30):                       # decline, alternating bars so RSI stays defined
        prev, px = px, px - 0.5 - (0.4 if i % 2 else -0.4)
        opens.append(prev); closes.append(px)
        highs.append(max(prev, px) + 0.2); lows.append(min(prev, px) - 0.2)
    prev, px = px, px - 1.0                   # the down bar the engulfer reverses
    opens.append(prev); closes.append(px); highs.append(prev + 0.2); lows.append(px - 0.3)
    engulf_open, engulf_close = px - 0.1, prev + 0.5   # outside bar, closes up, dominant lower wick
    opens.append(engulf_open); closes.append(engulf_close)
    highs.append(prev + 0.6); lows.append(px - 2.5)
    px = engulf_close
    for _ in range(rally_bars):
        prev, px = px, px + rally_step
        opens.append(prev); closes.append(px); highs.append(px + 0.2); lows.append(prev - 0.2)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1e6},
        index=pd.bdate_range("2024-01-01", periods=len(opens)),
    )


def bars(mask):
    return list(np.flatnonzero(np.asarray(mask, dtype=bool)))


# The engulfer is bar 31 in every fixture; a short RSI window is used so the
# confirmation can arrive before price has moved far, which is the case the
# drift cap is meant to admit.
gentle = build(0.2)
assert bars(_outside_bar_signals(gentle, 1.0, 0.0, False, False)[0]) == [31]

r = ind.rsi(gentle["Close"], 5)
first_cross = next(j for j in range(32, 52) if r.iloc[j] > 50)
assert first_cross == 35, first_cross
print("1. fixture: engulfer at 31, RSI recrosses 50 at 35  OK")

# 2. Entry lands on the confirming bar, not the engulfer.
signals = engulfer_rsi_confirmed(gentle, rsi_window=5)
assert bars(signals["entry"]) == [35], bars(signals["entry"])
print("2. entry is the confirmation bar, not the signal bar  OK")

# 3. Explosive rally: the confirmation arrives on the very next bar, but
#    price has already gapped ~7% past the engulfer, so the drift cap skips
#    it -- and dropping the cap takes it.
explosive = build(6.0)
assert bars(engulfer_rsi_confirmed(explosive, rsi_window=5)["entry"]) == []
uncapped = engulfer_rsi_confirmed(explosive, rsi_window=5, max_drift_pct=None)
assert bars(uncapped["entry"]), "uncapped rule should still take the late entry"
print("3. drift cap skips an entry that has already run away  OK")

# 4. A cap generous enough to cover that move admits the same entry, so it is
#    the threshold doing the work rather than the confirmation failing.
wide = engulfer_rsi_confirmed(explosive, rsi_window=5, max_drift_pct=50.0)
assert bars(wide["entry"]) == bars(uncapped["entry"])
print("4. widening the cap admits it again  OK")

# 5. No rally at all: RSI never recrosses, so there is no entry.
flat = build(0.0)
assert bars(engulfer_rsi_confirmed(flat, rsi_window=5)["entry"]) == []
print("5. no confirmation within max_wait -> no entry  OK")

# 6. Only the FIRST cross counts: a short max_wait that ends before the
#    confirmation must produce nothing.
assert bars(engulfer_rsi_confirmed(gentle, rsi_window=5, max_wait=2)["entry"]) == []
print("6. max_wait bounds the confirmation window  OK")

# 7. The position is held hold_bars and then closed.
held = run_backtest(gentle, engulfer_rsi_confirmed(gentle, rsi_window=5, hold_bars=10))
pos = pd.Series(range(len(gentle)), index=gentle.index)
assert [pos[t.exit_date] - pos[t.entry_date] for t in held] == [10]
print("7. hold_bars closes the position on schedule  OK")

print("\nall engulfer-RSI-confirmation checks passed")
