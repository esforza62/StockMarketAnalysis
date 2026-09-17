"""run_backtest used to DROP the position left open at the last bar.

That silently removed a strategy's most recent entries in proportion to
how long it holds, and the removal had a direction: a dip-and-recovery
position closes when it recovers, so the trades that closed were the
winners and the ones still open were losers in progress. It inflated both
the win rate and the average return, hardest for exactly the long-hold
strategies that top the regime desk.

These checks pin the fix and, just as importantly, pin that CLOSED trades
did not move: the change has to be purely additive or every stored figure
and every published table silently changes meaning.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import db as db_module
from smc_regime.backtest import Trade, run_backtest
from smc_regime.regime_backtest import summarize_by_regime

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


def frame(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
         "Close": closes, "Volume": [1_000] * len(closes)}, index=idx)


def signals(entries, exits, index):
    return pd.DataFrame({"entry": entries, "exit": exits}, index=index)


# Enter on bar 1, never exit: the position is open at the last bar.
df = frame([100.0, 110.0, 120.0, 130.0])
sig = signals([False, True, False, False], [False] * 4, df.index)
trades = run_backtest(df, sig)

check("1. the position open at the last bar is recorded, not dropped", len(trades) == 1)
check("2. it is flagged is_open", trades[0].is_open is True)
check("3. it is marked to market at the last bar's close",
      trades[0].exit_price == 130.0 and trades[0].exit_date == df.index[-1])
check("4. its return is measured from entry to that mark",
      abs(trades[0].return_pct - (130.0 / 110.0 - 1) * 100) < 1e-9)

# A trade that closes normally is untouched and NOT flagged.
sig2 = signals([False, True, False, False], [False, False, True, False], df.index)
closed = run_backtest(df, sig2)
check("5. a normally-closed trade is not flagged open",
      len(closed) == 1 and closed[0].is_open is False and closed[0].exit_price == 120.0)

# Entering and exiting repeatedly, ending flat, adds nothing.
df3 = frame([100.0, 110.0, 120.0, 130.0, 140.0])
sig3 = signals([False, True, False, True, False], [False, False, True, False, True], df3.index)
flat = run_backtest(df3, sig3)
check("6. a run that ends flat records no open trade",
      len(flat) == 2 and not any(t.is_open for t in flat))

check("7. at most one trade can be open, since the engine holds one position",
      sum(1 for t in run_backtest(df, sig) if t.is_open) == 1)

# ---- the summary keeps realised figures realised -------------------------
rows = []
for i in range(10):                      # ten closed winners at +10%
    rows.append({"ticker": "T", "strategy": "slow", "regime": "trending", "direction": "down",
                 "entry_date": pd.Timestamp("2024-01-01"), "exit_date": pd.Timestamp("2024-06-01"),
                 "return_pct": 10.0, "win": True, "size": 1.0, "is_open": False})
for i in range(10):                      # ten open losers at -30%
    rows.append({"ticker": "T", "strategy": "slow", "regime": "trending", "direction": "down",
                 "entry_date": pd.Timestamp("2024-07-01"), "exit_date": pd.Timestamp("2024-12-01"),
                 "return_pct": -30.0, "win": False, "size": 1.0, "is_open": True})
summary = summarize_by_regime(pd.DataFrame(rows)).iloc[0]

check("8. realised figures see only the closed trades",
      summary["trade_count"] == 10 and abs(summary["avg_return_pct"] - 10.0) < 1e-9
      and abs(summary["win_rate"] - 100.0) < 1e-9)

check("9. the open positions are counted separately",
      summary["open_trade_count"] == 10)

check("10. the unbiased figures include them",
      abs(summary["avg_return_all_pct"] - (-10.0)) < 1e-9
      and abs(summary["win_rate_all"] - 50.0) < 1e-9)

check("11. the gap between the pair is the size of the old bias",
      abs((summary["avg_return_pct"] - summary["avg_return_all_pct"]) - 20.0) < 1e-9)

# ---- the column survives a round trip through the database ---------------
import tempfile
tmp = Path(tempfile.mkdtemp()) / "t.db"
conn = db_module.connect(tmp)
frame_in = pd.DataFrame([
    {"ticker": "AAA", "strategy": "slow", "regime": "trending", "direction": "down",
     "entry_date": pd.Timestamp("2024-01-01", tz="UTC"), "exit_date": pd.Timestamp("2024-02-01", tz="UTC"),
     "return_pct": 5.0, "win": True, "size": 1.0, "is_open": True},
    {"ticker": "AAA", "strategy": "slow", "regime": "trending", "direction": "down",
     "entry_date": pd.Timestamp("2024-01-01", tz="UTC"), "exit_date": pd.Timestamp("2024-02-01", tz="UTC"),
     "return_pct": 5.0, "win": True, "size": 1.0, "is_open": False},
])
db_module.write_trades(conn, "2026-01-01T00:00:00+00:00", "1d", frame_in)
stored = sorted(r[0] for r in conn.execute("SELECT is_open FROM trades"))
check("12. is_open round-trips through the database", stored == [0, 1])

# An older database with no is_open column gains one rather than erroring.
legacy = Path(tempfile.mkdtemp()) / "old.db"
raw = sqlite3.connect(legacy)
raw.executescript("""
CREATE TABLE trades (run_id INTEGER, ticker TEXT, strategy_id INTEGER, regime TEXT,
                     direction TEXT, entry_date INTEGER, exit_date INTEGER,
                     return_pct REAL, win INTEGER, size REAL);
INSERT INTO trades VALUES (1,'AAA',1,'trending','down',100,200,5.0,1,1.0);
""")
raw.commit(); raw.close()
migrated = db_module.connect(legacy)
cols = {r[1] for r in migrated.execute("PRAGMA table_info(trades)")}
legacy_value = migrated.execute("SELECT is_open FROM trades").fetchone()[0]
check("13. a legacy database is migrated, with old rows left NULL",
      "is_open" in cols and legacy_value is None)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
