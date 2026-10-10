"""The grade's price-only components, reconstructed for every historical bar.

WHY THIS EXISTS. The setup grade can only be evaluated on dates it was
actually captured -- 24 of them, growing one per day -- because two of its
eight components cannot be dated backwards. `valuation` needs a forward P/E
as it stood on a past day and Yahoo serves only the current one; this is
why grade_backfill scores it at neutral half credit. `sector_industry`
needs peer direction counts taken from the regime store at capture time.

With 2-week forward windows, 24 overlapping dates is one or two effectively
independent observations. Every question about the scorer -- does RSI
really rank backwards, do the cuts still work, which component earns its
weight -- returns an answer that cannot be distinguished from noise, and no
amount of care in the statistics fixes a sample that small. It grows by one
date per day.

The other six components are computed from price alone:

    trend structure 20 + RSI 15 + MACD 15 + volume 15
      + regime streak 10 + daily/weekly alignment 10  =  85 of 100 points

Price history we have. Reconstructing those six across the cached history
turns 24 dates into roughly 1,900 and 9,940 rows into several hundred
thousand -- the sample the volume-ladder work ran on, where an information
coefficient is interpretable and a result can actually settle something.

IT CALLS THE REAL SCORER. Every component here is produced by the same
functions setup_score uses in production (`_ma_points`, `_rsi_points`, and
so on), driven by a vectorised restatement of `technicals.technical_snapshot`
evaluated at every bar instead of only the last one. A reimplementation
would measure a scorer that is not the deployed one, which is the failure
mode this module exists to avoid, so the private names are imported
deliberately rather than copied.

NO LOOK-AHEAD, and it is easy to get wrong here. Every input is a trailing
rolling or ewm statistic of bars at or before the bar it is stamped on. The
weekly rung is the last CLOSED weekly bar, shifted by one so the week in
progress cannot inform a day inside it. `verify_against_snapshot()` checks
the reconstruction against `technicals.technical_snapshot` on the final bar
-- where the two must agree exactly, because the snapshot is what the
nightly stored.

WHAT IT IS NOT. 85 points, not 100: conclusions drawn here transfer to
these six components and say nothing about valuation or sector alignment.
A price-only score is also not the deployed grade and its cuts do not
apply -- it is an instrument for measuring the scorer, not a replacement
for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import regime as rg
from .indicators import macd, rsi
from .setup_score import (
    _alignment_points,
    _ma_points,
    _macd_points,
    _rsi_points,
    _streak_points,
    _volume_points,
)
from .technicals import (
    _MA_FAST,
    _MA_SLOW,
    _VOLUME_BASELINE_BARS,
    _VOLUME_RECENT_BARS,
)

#: Max points, mirroring setup_score. Price-only: 85 of the grade's 100.
COMPONENT_MAX = {
    "trend_structure": 20, "rsi": 15, "macd": 15,
    "volume": 15, "streak": 10, "alignment": 10,
}
PRICE_ONLY_MAX = sum(COMPONENT_MAX.values())


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """`technicals.technical_snapshot` evaluated at EVERY bar.

    Deliberately mirrors that function field for field and window for
    window; if the snapshot changes, this has to change with it, and
    `verify_against_snapshot` is what catches the drift.
    """
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["rsi"] = rsi(close)
    out["ma_fast"] = close.rolling(_MA_FAST).mean()
    out["ma_slow"] = close.rolling(_MA_SLOW).mean()
    hist = macd(close)["histogram"]
    out["macd_hist"] = hist
    out["macd_hist_prev"] = hist.shift(1)
    if "Volume" in df.columns:
        vol = df["Volume"]
        recent = vol.rolling(_VOLUME_RECENT_BARS).mean()
        baseline = vol.rolling(_VOLUME_BASELINE_BARS).mean()
        out["volume_ratio"] = (recent / baseline).where(baseline > 0)
        # price move across the SAME recent window, so ratio and direction
        # are read together exactly as the snapshot intends
        out["price_change_pct"] = close.pct_change(_VOLUME_RECENT_BARS) * 100
    else:
        out["volume_ratio"] = np.nan
        out["price_change_pct"] = np.nan
    return out


def _streak_series(confirmed: pd.DataFrame) -> pd.Series:
    """Vectorised `regime_streak_bars`: run length of the (regime,
    direction) pair ending at each bar."""
    key = confirmed["regime"].astype(str) + "|" + confirmed["direction"].astype(str)
    grp = (key != key.shift()).cumsum()
    return key.groupby(grp).cumcount().add(1)


def _weekly_regime(df: pd.DataFrame, confirm_bars: int = 3) -> pd.DataFrame:
    """Confirmed weekly regime of the last week to CLOSE before each day.

    Two failure directions, and the asof below is the only form that avoids
    both. Reading the week a day sits inside is look-ahead of up to four
    bars. But the obvious fix -- shift(1) then forward-fill -- lags TWICE
    for any day that is not a Friday, because the nearest weekly label at
    or before a Monday is already last Friday and the shift pushes it back
    another week; most days then read a regime up to two weeks stale, which
    needlessly weakens the one component built on it.
    """
    wk = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum"}).dropna()
    if len(wk) < 30:
        return pd.DataFrame({"regime": "n/a", "direction": "n/a"}, index=df.index)
    wr = rg.confirmed_regime(rg.classify_regime(wk), confirm_bars=confirm_bars)
    wr = wr[["regime", "direction"]]
    # STRICTLY-BEFORE asof, not shift(1)+ffill. Both are safe, but shifting
    # then forward-filling lags twice for any day that is not a Friday: the
    # nearest weekly label at or before a Monday is already last Friday, and
    # the shift pushes it back another week, so most days read a regime up
    # to two weeks stale. searchsorted with side="left" takes the last week
    # that ENDED before this day -- one lag, never zero, never two.
    pos = wr.index.searchsorted(df.index, side="left") - 1
    out = pd.DataFrame({"regime": "n/a", "direction": "n/a"}, index=df.index)
    ok = pos >= 0
    out.loc[ok, "regime"] = wr["regime"].to_numpy()[pos[ok]]
    out.loc[ok, "direction"] = wr["direction"].to_numpy()[pos[ok]]
    return out


def price_components(df: pd.DataFrame, confirm_bars: int = 3) -> pd.DataFrame:
    """Per-bar price-only component points for one ticker."""
    ind = indicator_frame(df)
    daily = rg.confirmed_regime(rg.classify_regime(df), confirm_bars=confirm_bars)
    streak = _streak_series(daily)
    weekly = _weekly_regime(df, confirm_bars)

    n = len(df)
    cols = {c: np.full(n, np.nan) for c in COMPONENT_MAX}
    d_reg = daily["regime"].to_numpy()
    d_dir = daily["direction"].to_numpy()
    w_reg = weekly["regime"].to_numpy()
    w_dir = weekly["direction"].to_numpy()
    arr = {c: ind[c].to_numpy() for c in ind.columns}
    stk = streak.to_numpy()

    def _f(x):
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    for i in range(n):
        cols["trend_structure"][i] = _ma_points(
            _f(arr["close"][i]), _f(arr["ma_fast"][i]), _f(arr["ma_slow"][i]))[0]
        cols["rsi"][i] = _rsi_points(_f(arr["rsi"][i]), d_reg[i])[0]
        cols["macd"][i] = _macd_points(
            _f(arr["macd_hist"][i]), _f(arr["macd_hist_prev"][i]))[0]
        cols["volume"][i] = _volume_points(
            _f(arr["volume_ratio"][i]), _f(arr["price_change_pct"][i]))[0]
        cols["streak"][i] = _streak_points(int(stk[i]))
        cols["alignment"][i] = _alignment_points(
            {"regime": d_reg[i], "direction": d_dir[i]},
            {"regime": w_reg[i], "direction": w_dir[i]})[0]

    out = pd.DataFrame(cols, index=df.index)
    out["price_score"] = out[list(COMPONENT_MAX)].sum(axis=1)
    out["regime"] = d_reg
    out["direction"] = d_dir
    out["close"] = ind["close"]
    return out


def verify_against_snapshot(df: pd.DataFrame, confirm_bars: int = 3) -> dict:
    """Reconstruction vs the deployed path, on the final bar.

    `technicals.technical_snapshot` is what the nightly actually stored, so
    the last row of `indicator_frame` must reproduce it. A mismatch means
    the two have drifted and every number downstream describes a scorer
    that is not in production.
    """
    from .technicals import technical_snapshot

    snap = technical_snapshot(df)
    last = indicator_frame(df).iloc[-1]
    checked = ["close", "rsi", "ma_fast", "ma_slow", "macd_hist",
               "macd_hist_prev", "volume_ratio", "price_change_pct"]
    diffs = {}
    for k in checked:
        a, b = snap.get(k), last.get(k)
        if a is None and (b is None or pd.isna(b)):
            continue
        if a is None or b is None or pd.isna(b):
            diffs[k] = (a, None if b is None or pd.isna(b) else float(b))
        elif abs(float(a) - float(b)) > max(1e-6, abs(float(a)) * 1e-9):
            diffs[k] = (float(a), float(b))
    return {"ok": not diffs, "mismatches": diffs, "fields": len(checked)}


def reconstruct(prices: dict[str, pd.DataFrame], min_bars: int = 260,
                confirm_bars: int = 3, progress: bool = False) -> pd.DataFrame:
    """Long panel of price-only components for a universe."""
    frames = []
    for i, (ticker, df) in enumerate(prices.items()):
        if df is None or len(df) < min_bars:
            continue
        try:
            c = price_components(df, confirm_bars=confirm_bars)
        except Exception:
            continue
        c["ticker"] = ticker
        frames.append(c.reset_index().rename(columns={"index": "date", "Date": "date"}))
        if progress and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(prices)}", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
