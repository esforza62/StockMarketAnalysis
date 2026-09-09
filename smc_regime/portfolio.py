"""Portfolio-level performance statistics for a strategy's trade log.

regime_backtest.py summarises trades one at a time -- win rate, average
return per trade, per-ticker compounding. None of that is a Sharpe ratio:
risk-adjusted return needs an equity curve, which needs to know what was
held on every single day, not just what each trade eventually returned.

The construction here is the simplest defensible one:

  * a trade entered at the close of day i and exited at the close of day j
    earns that ticker's close-to-close return on days i+1 .. j;
  * capital is split equally across whatever is open on a given day, and
    sits in cash earning the risk-free rate when nothing is open.

That "equal weight across concurrently open positions" rule implies daily
rebalancing between open names and full investment whenever at least one
position is open, so a strategy that is usually holding one lonely position
looks more volatile than one holding thirty. `exposure` and
`avg_open_positions` in the stats are there to make that visible rather
than hidden -- a Sharpe computed off two concurrent positions is a
different animal from one computed off forty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trade_daily_returns(
    df: pd.DataFrame,
    trades,
    side: str = "long",
    cost_bps_per_side: float = 0.0,
) -> pd.Series:
    """One ticker's daily return contribution, NaN on days with no position.

    Costs are charged as a return haircut on the entry day and again on the
    exit day, which is where the spread and commission are actually paid.

    SHORTS CARRY AN ASSUMPTION THAT CHANGES THE ANSWER. A short's daily
    contribution here is -1 x the price return, which is a short re-marked to
    a constant notional every day -- the only convention consistent with
    portfolio_returns() splitting capital equally across open positions each
    day. It is not "short and hold":

      * it caps the damage from a name that runs away, where a real static
        short compounds against you without limit;
      * it collects the volatility drag of whatever it is short, which on
        high-volatility names is a large positive unrelated to the signal;
      * it requires trading every position every day, which the entry/exit
        cost model here does not charge for.

    So portfolio statistics for a short book describe a daily-rebalanced
    strategy, and should be read next to the per-trade static P&L, which is
    what a real held short would have made. On the mirror short of
    rsi_dip_recovery those two disagree completely: the daily-rebalanced book
    shows a positive hedged return, while the trades themselves average
    -6.5% each with a worst case of -2,628%.

    A genuinely static short book needs per-position capital accounting
    rather than equal-weighted returns -- a larger change than this module,
    and not attempted here rather than approximated badly.

    For LONGS the distinction does not exist: compounding a position's own
    daily returns reproduces the trade's total return either way.
    """
    close = df["Close"]
    bar_return = close.pct_change()
    positions = pd.Series(np.nan, index=df.index)
    cost = cost_bps_per_side / 10_000.0
    sign = 1.0 if side == "long" else -1.0

    index_pos = pd.Series(range(len(df)), index=df.index)
    for trade in trades:
        if trade.entry_date not in index_pos.index or trade.exit_date not in index_pos.index:
            continue
        start, end = index_pos[trade.entry_date], index_pos[trade.exit_date]
        if end <= start:
            continue
        held = df.index[start + 1 : end + 1]
        # A second trade never overlaps the first on one ticker (the engine
        # is single-position), so writing rather than accumulating is safe.
        positions.loc[held] = sign * bar_return.loc[held]
        positions.loc[held[0]] -= cost
        positions.loc[held[-1]] -= cost
    return positions


def portfolio_returns(per_ticker: dict[str, pd.Series], rf_annual: float = 0.0) -> pd.DataFrame:
    """Equal-weight the open positions each day; hold cash when flat.

    Returns a frame with the daily portfolio return, how many positions were
    open, and whether the day was invested at all.
    """
    panel = pd.DataFrame(per_ticker)
    open_count = panel.notna().sum(axis=1)
    daily = panel.mean(axis=1, skipna=True)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    daily = daily.where(open_count > 0, rf_daily)
    return pd.DataFrame({"return": daily, "open_positions": open_count, "invested": open_count > 0})


def hedged_returns(portfolio: pd.DataFrame, benchmark: pd.Series, side: str = "long") -> pd.Series:
    """Strip the market out: the portfolio's return minus the benchmark's on
    every invested day, and zero on days the strategy holds cash.

    This is the portfolio-level counterpart of the study's excess-over-random-
    timing measure, and it is the only fair way to read the SHORT rules. Every
    bearish result in docs/OUTSIDE_BAR_RSI.md is a statement about a name
    UNDERPERFORMING, not about it falling: a naked short also pays away the
    market's own drift, which over this sample was about 16%/year. Hedging
    asks the question the study actually answered.

    A beta of 1 is assumed rather than estimated -- crude, but it avoids
    fitting a beta per name on the same sample being measured.

    The hedge is sign-aware, and it has to be: a market-neutral SHORT book is
    short the name and LONG the benchmark, so the benchmark return is added
    back, not subtracted. Subtracting on both sides (the obvious mistake)
    double-counts the market against the shorts and turns a roughly
    market-neutral book into a 2x bearish one.
    """
    aligned = benchmark.reindex(portfolio.index)
    beta_sign = 1.0 if side == "long" else -1.0
    diff = portfolio["return"] - beta_sign * aligned
    return diff.where(portfolio["invested"], 0.0)


def market_model(returns: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0) -> dict:
    """Regress the strategy on the benchmark: beta, annualised alpha, and the
    information ratio of what is left over.

    hedged_returns() assumes a beta of exactly 1, which flatters any portfolio
    that holds higher-beta names than the index: in a rising market the
    leftover market exposure shows up as alpha. This estimates beta from the
    data instead, so alpha is what survives after the portfolio's ACTUAL
    market sensitivity is removed. The information ratio is the Sharpe of the
    residuals -- risk-adjusted skill, with the market taken out.

    Beta is fitted on the same sample it is then used to adjust, which is
    mildly optimistic; with ~1,250 daily observations the fit is stable
    enough that this matters much less than the beta-1 assumption it fixes.
    """
    r = returns.dropna()
    b = benchmark.reindex(r.index).dropna()
    r = r.reindex(b.index)
    if len(r) < 30 or b.var(ddof=1) == 0:
        return {}
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    r_ex, b_ex = r - rf_daily, b - rf_daily

    beta = b_ex.cov(r_ex) / b_ex.var(ddof=1)
    alpha_daily = r_ex.mean() - beta * b_ex.mean()
    residual = r_ex - beta * b_ex
    resid_sd = residual.std(ddof=1)
    return {
        "beta": beta,
        "alpha_ann_pct": ((1 + alpha_daily) ** TRADING_DAYS - 1) * 100,
        "information_ratio": alpha_daily / resid_sd * np.sqrt(TRADING_DAYS) if resid_sd > 0 else np.nan,
    }


def performance_stats(
    returns: pd.Series,
    rf_annual: float = 0.0,
    open_positions: pd.Series | None = None,
) -> dict:
    """Sharpe, Sortino, and the context needed to read them honestly.

    Sortino replaces total volatility with downside deviation -- the RMS of
    the negative excess returns only -- so a strategy is not penalised for
    upside surprises. It is computed over the full sample (zeros included in
    the mean-square, not dropped), which is the convention that keeps it
    comparable across strategies with different trade frequencies; dividing
    only by the count of down days instead would flatter a rarely-losing
    strategy purely for trading less.

    A consequence worth knowing before reading the output: under this
    convention downside deviation can never exceed total volatility, so
    Sortino >= Sharpe for any positive-mean series. "Sortino beats Sharpe"
    therefore means nothing on its own -- it is the RATIO between them that
    says whether a strategy's volatility is mostly upside (ratio well above
    1) or mostly drawdown (ratio near 1).
    """
    r = returns.dropna()
    if r.empty:
        return {}
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = r - rf_daily

    ann_return = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mean_excess = excess.mean()

    sd = excess.std(ddof=1)
    sharpe = mean_excess / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan

    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    sortino = mean_excess / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    years = len(r) / TRADING_DAYS
    # Standard error of a Sharpe estimate (Lo 2002), ignoring autocorrelation:
    # even a "good" Sharpe is indistinguishable from zero on a short sample.
    se = np.sqrt((1 + 0.5 * sharpe ** 2) / years) if np.isfinite(sharpe) else np.nan

    stats = {
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_se": se,
        "sharpe_t": sharpe / se if se and np.isfinite(se) else np.nan,
        "years": years,
    }
    if open_positions is not None:
        aligned = open_positions.reindex(r.index)
        stats["exposure_pct"] = (aligned > 0).mean() * 100
        stats["avg_open_positions"] = aligned[aligned > 0].mean()
    return stats
