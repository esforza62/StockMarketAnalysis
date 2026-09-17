"""Event-driven long-only backtest: turns a strategy's entry/exit signals into a trade log."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import STRATEGIES

# Realised-volatility window for vol_target_sizes(). 60 bars is long enough
# that the estimate is not dominated by a single gap, short enough to track
# a regime change in weeks rather than quarters.
_VOL_WINDOW = 60

# Bars per year, PER INTERVAL. Annualising a volatility means scaling by the
# square root of how many bars fit in a year, so a single constant is only
# ever right for one bar size. Using the daily 252 everywhere understated
# hourly vol by 2.55x and 15-minute vol by 5.10x -- which drove almost every
# intraday position to the 1.0 cap and made sizing inert on those intervals
# -- while overstating weekly vol by 2.20x and shrinking weekly positions to
# roughly a third for no reason. Observed on the first nightly that computed
# sizes for all four: mean size 0.992 on 15m and 0.949 on 1h against 0.665
# on 1d, with 1w down at 0.385.
#
# US equities: ~252 sessions, 6.5 hours each, 26 fifteen-minute bars each.
_BARS_PER_YEAR = {
    "1d": 252,
    "1h": 252 * 6.5,
    "15m": 252 * 26,
    "1w": 52,
}
_DEFAULT_BARS_PER_YEAR = 252


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    # Fraction of a full position held, fixed at entry. 1.0 is the
    # unweighted behaviour every existing caller gets by default.
    size: float = 1.0
    # True when the position was still open at the last bar and has been
    # marked to market there rather than closed by the strategy's own
    # rule. Defaults False, so every trade built the old way is a real
    # exit and nothing that reads Trade today changes meaning.
    is_open: bool = False

    @property
    def return_pct(self) -> float:
        """What the ASSET did, independent of how much of it was held.

        Deliberately unaffected by `size`: this is the number every
        existing summary, the regime tables and the stored trade rows all
        mean by "return", and silently rescaling it would change the
        meaning of published figures rather than add a new one.
        """
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def weighted_return_pct(self) -> float:
        """The contribution to EQUITY -- the asset's move times the
        fraction held. Identical to return_pct at full size, so an equity
        curve built on this is correct whether or not sizing is in use."""
        return self.return_pct * self.size


def vol_target_sizes(
    df: pd.DataFrame,
    target_vol_pct: float,
    window: int = _VOL_WINDOW,
    cap: float = 1.0,
    interval: str = "1d",
) -> pd.Series:
    """Per-bar position fraction that equalises RISK rather than capital.

    min(cap, target / trailing_realised_vol), computed from closes up to
    and including each bar -- the value read at an entry bar uses no data
    from that trade's own future.

    This exists because rsi_dip_recovery's drawdown is not a signal
    problem. Bucketing its 2075 trades by trailing 60-day vol at entry,
    the LOSS RATE is flat across quintiles (60/75/87/84/80 losing trades)
    and the median losing hold is ~420 days in every bucket -- but the
    average loss runs -10.6% on the calmest fifth against -38.1% on the
    wildest. The strategy is not wrong more often on volatile names; it is
    wrong by the same amount of time and a much larger amount of money.

    That is why entry filters cost so much here. The wildest vol quintile
    is also the most PROFITABLE (+27.3%/trade against +9.6% on the
    calmest, and a third of all gross profit), so excluding it to remove
    the risk removes the edge with it. Scaling the position instead leaves
    the trade set exactly as it was and changes only the stake.

    Measured through this engine over 409 tickers from 2019, per-ticker
    equity on weighted_return_pct, against the trend-filtered variant on
    identical bars and identical equity maths:

        variant                  tickers  trades  meanDD   worst  %<-50  medTot
        rsi_dip_recovery             409    2626   -21.7   -99.2   14.2   100.2
        rsi_dip_trend_filter         308    1002    -7.7   -86.3    3.2    45.0
        + vol-target 20%             409    2626   -12.2   -71.9    1.5    70.8

    The sized baseline is better than the trend filter on every risk
    measure AND keeps 57% more of the edge, on the full universe rather
    than the two-thirds of it the filter still trades. The two compose
    (trend filter + 20% target reaches -62.0 worst) but at medTot 37.5,
    which is paying for risk twice.

    `interval` selects the annualisation factor and is not optional in
    spirit: the same 60 bars mean nine trading days on hourly data and
    sixty weeks on weekly, so scaling both by the daily 252 produces a
    number that is not a volatility at all.

    A cap of 1.0 means this can only ever reduce a position, never lever
    one up: a very calm name is held at full size, not at three times it.
    Scaling UP a calm name would be the natural next step and is
    deliberately not taken here -- it would add leverage to a result whose
    whole claim is that it reduces risk without touching the signal.
    """
    if target_vol_pct <= 0:
        raise ValueError("target_vol_pct must be positive")
    returns = df["Close"].pct_change()
    bars_per_year = _BARS_PER_YEAR.get(interval, _DEFAULT_BARS_PER_YEAR)
    vol = returns.rolling(window).std() * np.sqrt(bars_per_year) * 100
    sizes = (target_vol_pct / vol).clip(upper=cap)
    # Before `window` bars there is no estimate. Full size is the honest
    # default -- it is what the engine did before sizing existed, so an
    # early trade is not silently shrunk by a number that does not exist.
    return sizes.fillna(cap)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
    size_series: pd.Series | None = None,
) -> list[Trade]:
    """Simulate a single-position long-only strategy from entry/exit signals.

    stop_loss_pct, if set, closes the position at entry_price * (1 -
    stop_loss_pct/100) the first bar whose Low touches that level --
    checked ahead of that same bar's own exit signal, since a stop is a
    risk-management floor, not a strategy read on the bar's close.

    stop_loss_pct_series is the same idea but per-entry rather than one
    fixed percentage for every trade -- e.g. an ATR-based stop, where a
    volatile ticker gets a wider stop than a calm one. Looked up at the
    entry bar's date to fix that trade's own stop percentage for its
    whole duration (not re-computed every bar). Takes precedence over
    stop_loss_pct if both are given.

    max_hold_bars, if set, force-closes the position at that bar's Close
    once it has been held this many bars without hitting its own exit
    signal or stop -- caps how long a trade can sit waiting for an exit
    condition that may never come (rsi_dip_recovery's overbought exit, in
    particular, has no guarantee of ever firing if a dip just keeps
    falling).

    size_series, if set, is the fraction of a full position to hold, read
    at the entry bar and fixed for that trade's whole duration -- see
    vol_target_sizes(). It changes NOTHING about which trades are taken or
    when they close: the entry and exit logic never consults it, so a
    sized run and an unsized one produce the same trades in the same
    order, differing only in Trade.size. That is the point of putting it
    here rather than in a strategy -- a filter changes the sample and can
    flatter itself by dropping losers, whereas a weight cannot.

    Neither a stop nor the time limit is ever checked on the entry bar
    itself: in_position only becomes True after that iteration's checks
    already ran, so nothing can close a position before it exists.
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    stop_price = None
    bars_held = 0
    size = 1.0

    for date, row in signals.iterrows():
        close = df.loc[date, "Close"]

        if in_position:
            bars_held += 1
            if stop_price is not None:
                low = df.loc[date, "Low"]
                if low <= stop_price:
                    trades.append(Trade(entry_date, date, entry_price, stop_price, size))
                    in_position = False
                    continue
            if max_hold_bars is not None and bars_held >= max_hold_bars:
                trades.append(Trade(entry_date, date, entry_price, close, size))
                in_position = False
                continue

        if not in_position and row["entry"]:
            in_position = True
            entry_date = date
            entry_price = close
            bars_held = 0
            if size_series is not None:
                raw = size_series.get(date)
                size = float(raw) if pd.notna(raw) else 1.0
            if stop_loss_pct_series is not None:
                pct = stop_loss_pct_series.get(date)
                stop_price = entry_price * (1 - pct / 100) if pd.notna(pct) else None
            elif stop_loss_pct is not None:
                stop_price = entry_price * (1 - stop_loss_pct / 100)
            else:
                stop_price = None
        elif in_position and row["exit"]:
            trades.append(Trade(entry_date, date, entry_price, close, size))
            in_position = False

    # MARK THE SURVIVOR TO MARKET instead of dropping it.
    #
    # This loop used to end here with the position still open, and that
    # trade was never appended -- so `trades` held closed trades only and
    # every strategy was missing its recent entries in proportion to how
    # long it holds. On the 1d run the final hold-window was missing ~72%
    # of rsi_dip_recovery's expected entries and ~88% of
    # rsi_dip_trend_filter's, against ~0% for the 5-12 day strategies.
    #
    # The bias is not just incompleteness, it has a direction. A
    # dip-and-recovery position closes when it recovers, so the ones that
    # close are the winners and the ones still open are losers in
    # progress. Dropping them inflates both the win rate and the average
    # return, and it does so hardest for exactly the strategies that top
    # the regime desk.
    #
    # The trade is recorded at the last bar's close and flagged is_open,
    # so a reader can have the unbiased set, the realised-only set, or
    # both. Nothing about a CLOSED trade changes: this appends after the
    # loop and touches no existing branch.
    if in_position:
        last_date = signals.index[-1]
        trades.append(
            Trade(entry_date, last_date, entry_price, df.loc[last_date, "Close"], size, is_open=True)
        )

    return trades


def backtest_strategy(
    df: pd.DataFrame,
    strategy: str,
    stop_loss_pct: float | None = None,
    stop_loss_pct_series: pd.Series | None = None,
    max_hold_bars: int | None = None,
    size_series: pd.Series | None = None,
    target_vol_pct: float | None = None,
    interval: str = "1d",
) -> list[Trade]:
    """target_vol_pct is the convenience form of size_series: it builds one
    from this ticker's own bars via vol_target_sizes(). Passing both is a
    contradiction rather than a precedence question, so it raises."""
    if size_series is not None and target_vol_pct is not None:
        raise ValueError("pass size_series or target_vol_pct, not both")
    if target_vol_pct is not None:
        size_series = vol_target_sizes(df, target_vol_pct, interval=interval)
    signals = STRATEGIES[strategy](df)
    # A strategy may carry its own per-trade invalidation. swing_failure
    # puts the stop under the swept low, because that is where the setup is
    # actually wrong -- a fixed percentage would be an arbitrary distance
    # from a structural level. Only strategies that emit the column get a
    # stop; the rest keep running stopless, so adding this changed nothing
    # about the existing nineteen.
    #
    # An explicit argument still wins, so a caller sweeping stop levels can
    # override the strategy's own without editing it.
    if (
        "stop_pct" in signals
        and stop_loss_pct is None
        and stop_loss_pct_series is None
    ):
        stop_loss_pct_series = signals["stop_pct"]

    return run_backtest(
        df, signals,
        stop_loss_pct=stop_loss_pct, stop_loss_pct_series=stop_loss_pct_series, max_hold_bars=max_hold_bars,
        size_series=size_series,
    )
