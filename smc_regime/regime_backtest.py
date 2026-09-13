"""Regime-conditioned backtest harness.

Backtests each candidate strategy on each ticker, then tags every individual
trade with the SMC regime (and direction) that was active on its *entry*
date -- not the ticker's current regime. Aggregating those tags answers
"which strategy performs best in which regime?" directly from trade-level
outcomes, rather than proxying regime with a whole-period snapshot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .backtest import backtest_strategy, vol_target_sizes
from .data import fetch_ohlcv
from .regime import RegimeThresholds, classify_regime, confirmed_regime, regime_streak_bars
from .split_guard import UNADJUSTED_INTERVALS, filter_contaminated_trades, load_split_cache
from .strategies import STRATEGIES
from .technicals import technical_snapshot

# Tiingo's historical-prices endpoint has no batch/multi-ticker mode --
# confirmed by testing directly against the API, it treats a comma-joined
# ticker string as one invalid symbol. Every ticker needs its own request,
# so fetching is network-latency-bound, not compute-bound: firing requests
# concurrently instead of waiting on each one sequentially is what actually
# fixes the wall-clock scaling problem as the tracking universe grows.
_FETCH_WORKERS = 16

# Annualised volatility the sized column targets. 20% is where the
# risk/edge trade-off measured on rsi_dip_recovery landed: worst per-ticker
# drawdown -99.2% -> -71.9% and tickers drawing past -50% 14.2% -> 1.5%,
# while keeping 71% of the median per-ticker total return. Lower targets
# keep cutting drawdown but start costing more edge than they save.
DEFAULT_TARGET_VOL_PCT = 20.0


def _fetch_all(tickers: list[str], period: str, interval: str, start_date: str | None) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {
            pool.submit(fetch_ohlcv, ticker, period=period, interval=interval, start_date=start_date): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                dfs[ticker] = future.result()
            except Exception:
                continue
    return dfs


def collect_trades(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    t: RegimeThresholds = RegimeThresholds(),
    start_date: str | None = None,
    confirm_bars: int = 3,
    target_vol_pct: float | None = DEFAULT_TARGET_VOL_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest every strategy on every ticker and tag each trade with the
    regime/direction active on its entry date.

    Regime is confirmed_regime()-filtered before tagging, not raw
    classify_regime() output: a trade tagged with a regime that was only
    true for a single noisy bar isn't a meaningful measurement of "which
    strategy works in this regime" -- confirm_bars requires the regime to
    have held for that many consecutive bars before a trade's entry is
    attributed to it. Set confirm_bars=1 to disable and get the raw
    per-bar behavior this used before.

    target_vol_pct records a volatility-targeted position size on every
    trade WITHOUT changing which trades are taken -- the engine never
    consults it for entry or exit (see backtest.run_backtest). The point is
    to carry both readings on one trade set: return_pct is the unsized
    result every existing figure means, and return_pct * size is the same
    trades run at a risk-equalised stake. Set it to None to record full
    size everywhere.

    Returns (trades, latest_regime): `latest_regime` is one row per ticker
    with its confirmed regime/direction/streak_bars as of the LAST bar in
    this run, plus that bar's technical_snapshot() readings (RSI, MACD
    histogram, volume participation, 50/200-MA position) -- a byproduct of
    the same per-ticker fetch and classification already computed for
    trade-tagging above, captured so downstream consumers (the setup score,
    a "what's every tracked ticker doing right now" dashboard) can read a
    ticker's current state straight from the DB instead of re-fetching and
    re-classifying live.
    """
    records = []
    latest_records = []

    dfs = _fetch_all(tickers, period, interval, start_date)
    for ticker in tickers:
        df = dfs.get(ticker)
        if df is None:
            continue
        regime = confirmed_regime(classify_regime(df, t), confirm_bars=confirm_bars)
        latest = regime.iloc[-1]
        latest_records.append(
            {
                "ticker": ticker,
                "regime": latest["regime"],
                "direction": latest["direction"],
                "streak_bars": regime_streak_bars(regime),
                # Read off the same already-fetched df rather than re-fetching
                # per ticker later -- the setup scorer runs over the whole
                # universe from the DB, and this is the only place the price
                # series is already in hand.
                **technical_snapshot(df),
            }
        )

        # One size series per ticker, reused across all 19 strategies --
        # it depends only on the price history, not on the signal.
        sizes = vol_target_sizes(df, target_vol_pct) if target_vol_pct else None
        for strategy in STRATEGIES:
            for trade in backtest_strategy(df, strategy, size_series=sizes):
                if trade.entry_date not in regime.index:
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "regime": regime.loc[trade.entry_date, "regime"],
                        "direction": regime.loc[trade.entry_date, "direction"],
                        "entry_date": trade.entry_date,
                        "exit_date": trade.exit_date,
                        "return_pct": trade.return_pct,
                        "win": trade.return_pct > 0,
                        # 1.0 unless this run used position sizing. Carried
                        # so the equity-curve figures (compounding and
                        # drawdown) can weight by it while the reported
                        # per-trade return stays the asset's own move.
                        "size": trade.size,
                    }
                )

    trades = pd.DataFrame.from_records(records)
    if interval.lower() in UNADJUSTED_INTERVALS:
        # Tiingo's IEX intraday endpoint has no adjusted-price equivalent
        # (unlike the daily EOD endpoint, which uses adjClose -- see
        # data.py), so a trade whose window straddles a real stock split
        # shows a fake near-total-wipeout return. Drop those before they
        # ever get written, rather than patching every downstream reader.
        trades = filter_contaminated_trades(trades, load_split_cache())
    return trades, pd.DataFrame.from_records(latest_records)


def _equity_returns(trades: pd.DataFrame, sized: bool = True) -> pd.Series:
    """One ticker's trades as EQUITY contributions, ordered by entry date.

    return_pct is what the asset did; multiplying by `size` gives what the
    account did. Runs predating position sizing have no size column, and a
    missing or null size means a full position -- so this reduces exactly
    to the unweighted series it replaces, and the historical figures it
    feeds are unchanged.

    sized=False ignores the column outright, which is how the unsized and
    sized columns are produced from ONE trade set: the trades are the same
    rows either way, so the pair is a like-for-like comparison rather than
    two backtests that might differ for other reasons.
    """
    ordered = trades.sort_values("entry_date")
    returns = ordered["return_pct"]
    if not sized or "size" not in ordered:
        return returns
    return returns * ordered["size"].fillna(1.0)


def _ticker_compounded_return_pct(trades: pd.DataFrame, sized: bool = True) -> float:
    """Sequential compounding of one ticker's own trades, ordered by entry date.

    Valid here because this backtest only ever holds one position per ticker
    at a time -- a single ticker's trades genuinely do happen one after
    another for that ticker's own capital.
    """
    returns = _equity_returns(trades, sized)
    growth = (1 + returns / 100).prod()
    return (growth - 1) * 100


def _compounded_return_pct(group: pd.DataFrame, sized: bool = True) -> float:
    """Equal-weighted average of each ticker's own compounded return.

    Compounding trades from DIFFERENT tickers together as if they were one
    sequential stream (the previous approach) is wrong: those trades happen
    concurrently in real time, not one after another, and chaining thousands
    of them as if serial explodes exponentially into meaningless numbers.
    Compounding within a ticker (realistic) and then averaging equally
    across tickers approximates splitting capital evenly across every
    ticker trading this strategy in this regime -- still a simplification,
    but not a nonsensical one.
    """
    per_ticker = group.groupby("ticker").apply(_ticker_compounded_return_pct, sized=sized)
    return per_ticker.mean()


def _ticker_max_drawdown_pct(trades: pd.DataFrame, sized: bool = True) -> float:
    """Peak-to-trough drawdown on one ticker's own sequential equity curve --
    same ordering/compounding basis as _ticker_compounded_return_pct, so a
    string of wins followed by one catastrophic loss shows up here even
    when the AVERAGE return still looks good."""
    returns = _equity_returns(trades, sized)
    equity = (1 + returns / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    return drawdown.min()


def _max_drawdown_pct(group: pd.DataFrame, sized: bool = True) -> float:
    """Worst per-ticker max drawdown across every ticker that traded this
    strategy in this regime bucket -- deliberately the WORST case, not an
    average across tickers, since averaging a tail-risk figure hides
    exactly the risk it exists to surface (a strategy where most tickers
    drew down 10% but one drew down 90% is not well-described by "50%
    average drawdown")."""
    per_ticker = group.groupby("ticker").apply(_ticker_max_drawdown_pct, sized=sized)
    return per_ticker.min()


def summarize_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level results into (regime, direction, strategy) performance.

    Trending and parabolic each split into "up" and "down" -- a long-only
    strategy set behaves very differently riding a trend up vs. trying to
    catch bounces in one going down, so collapsing direction away would
    hide exactly the asymmetry that matters. Choppy has no direction
    ("n/a") since it isn't trending either way.
    """

    def _agg(group: pd.DataFrame) -> pd.Series:
        returns = group["return_pct"]
        losers = returns[returns < 0]
        hold_days = (group["exit_date"] - group["entry_date"]).dt.total_seconds() / 86400
        return pd.Series(
            {
                "trade_count": len(group),
                "win_rate": (returns > 0).mean() * 100,
                "avg_return_pct": returns.mean(),
                "avg_hold_days": hold_days.mean(),
                "total_return_pct": returns.sum(),
                "compounded_return_pct": _compounded_return_pct(group, sized=False),
                "worst_trade_pct": returns.min(),
                "loss_rate_pct": (returns < 0).mean() * 100,
                "avg_loss_pct": losers.mean() if not losers.empty else 0.0,
                "max_drawdown_pct": _max_drawdown_pct(group, sized=False),
                # The same trades at a volatility-targeted stake. Only the
                # equity-curve figures get a sized twin: win rate, average
                # return and worst trade are properties of the trades
                # themselves and a weight cannot move them, which is the
                # whole reason sizing is comparable in the first place.
                "compounded_return_sized_pct": _compounded_return_pct(group, sized=True),
                "max_drawdown_sized_pct": _max_drawdown_pct(group, sized=True),
            }
        )

    summary = trades.groupby(["regime", "direction", "strategy"]).apply(_agg)
    summary["trade_count"] = summary["trade_count"].astype(int)
    return summary.reset_index().sort_values(["regime", "direction", "avg_return_pct"], ascending=[True, True, False])
