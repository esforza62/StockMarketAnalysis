"""Does a strategy's ENTRY TIMING beat doing nothing in the same conditions?

Every headline number in this project is a return, and a return in a rising
market mostly measures the market. This asks the harder question: for each
trade, what would you have made entering at an ARBITRARY bar in the same
regime and holding the same number of bars? The difference is the edge that
belongs to the strategy rather than to the tape.

The benchmark is deliberately not buy-and-hold from a fixed date. That is a
single entry, and in 2019-2026 it is a lucky one -- comparing against it
measures the start date as much as the strategy. Averaging over every bar in
the same regime removes the entry-date lottery, which is the only way "do
nothing" becomes a fair comparison rather than a different bet.

    python -m smc_regime.excess_cli --tickers 60 --strategies rsi,macd

WHAT IT FOUND, daily bars since 2019, 60 tickers: pooled across regimes not
one of six strategies showed a significant positive excess (best t = 1.05),
including rsi_dip_recovery at -0.74% -- it returns +14.3% per trade where
the same regime returns +15.0% to anyone at all. Split by regime, rsi in
choppy showed +7.78% excess (t = 3.93, 74% of trades positive) and donchian
in choppy -1.71% (t = -2.10): mean reversion works in chop and breakouts get
shredded by it. Seventeen cells were examined, so roughly one should look
significant by chance -- but t = 3.93 survives a Bonferroni correction, and
both effects have the sign theory predicts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .backtest import backtest_strategy
from .metadata import _load_cache as load_exchange_cache
from .regime import RegimeThresholds, classify_regime, confirmed_regime
from .strategies import STRATEGIES

_MIN_CELL_TRADES = 40

#: A NASDAQ name is measured against QQQ, everything else against SPY.
#: Comparing a stock to ITSELF held from a fixed date is not a neutral
#: benchmark -- tracking_universe.txt was assembled today, so "buy and hold
#: NVDA since 2019" quietly assumes you knew in 2019 which names to pick.
#: The index is the thing that was actually available to buy with no
#: foresight, which makes it the only honest floor for a strategy to clear.
_BENCHMARKS = {"NASDAQ": "QQQ"}
_DEFAULT_BENCHMARK = "SPY"


def benchmark_for(ticker: str, exchanges: dict[str, str] | None = None) -> str:
    exchanges = exchanges if exchanges is not None else load_exchange_cache()
    return _BENCHMARKS.get(exchanges.get(ticker.upper(), ""), _DEFAULT_BENCHMARK)


def compound_cagr(returns: list[float], years: float) -> float:
    """Sequential compounding of one ticker's trades, annualised -- the same
    basis regime_backtest._ticker_compounded_return_pct uses, since this
    engine only ever holds one position per ticker at a time."""
    if years <= 0:
        return float("nan")
    total = float((1 + pd.Series(returns) / 100).prod())
    return (total ** (1 / years) - 1) * 100 if total > 0 else -100.0


def _t_stat(per_ticker: pd.Series) -> float:
    """Clustered by ticker, not pooled over trades.

    Trades overlap in calendar time and a market-wide move hits every open
    position at once, so pooled trades are nowhere near independent draws
    and a pooled t-stat reads far too confident. One number per ticker is
    the conservative reading.
    """
    if len(per_ticker) < 3:
        return float("nan")
    return per_ticker.mean() / (per_ticker.std() / len(per_ticker) ** 0.5)


def trade_excess(df: pd.DataFrame, ticker: str, strategies: list[str], confirm_bars: int = 3) -> pd.DataFrame:
    """One row per trade: its return, its matched baseline, and the difference."""
    regime = confirmed_regime(classify_regime(df, RegimeThresholds()), confirm_bars=confirm_bars)
    label = (regime["regime"].astype(str) + "/" + regime["direction"].astype(str)).reindex(df.index)
    close = df["Close"].to_numpy()
    position = {stamp: i for i, stamp in enumerate(df.index)}

    rows = []
    for name in strategies:
        trades = backtest_strategy(df, name)
        if not trades:
            continue
        holds = sorted({position[t.exit_date] - position[t.entry_date] for t in trades if t.exit_date in position})
        # For each holding length actually used, the mean forward return of
        # every bar, grouped by that bar's regime. Computed once per length
        # rather than per trade -- the same lengths recur constantly.
        forward_means = {}
        for h in holds:
            if h <= 0:
                continue
            forward = pd.Series(
                np.concatenate([close[h:] / close[:-h] - 1, np.full(h, np.nan)]), index=df.index
            ) * 100
            forward_means[h] = forward.groupby(label).mean()

        for t in trades:
            h = position[t.exit_date] - position[t.entry_date]
            bucket = label.get(t.entry_date)
            if h <= 0 or bucket is None or h not in forward_means:
                continue
            baseline = forward_means[h].get(bucket, np.nan)
            if pd.isna(baseline):
                continue
            rows.append(
                {"strategy": name, "ticker": ticker, "regime": bucket, "bars_held": h,
                 "return_pct": t.return_pct, "baseline_pct": baseline,
                 "excess_pct": t.return_pct - baseline}
            )
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overall, by regime) -- both clustered by ticker."""
    def agg(group: pd.DataFrame) -> pd.Series:
        per_ticker = group.groupby("ticker")["excess_pct"].mean()
        return pd.Series({
            "trades": len(group), "tickers": group["ticker"].nunique(),
            "mean_return_pct": group["return_pct"].mean(),
            "baseline_pct": group["baseline_pct"].mean(),
            "excess_pct": group["excess_pct"].mean(),
            "t": _t_stat(per_ticker),
            "pct_positive": (group["excess_pct"] > 0).mean() * 100,
        })

    overall = trades.groupby("strategy").apply(agg, include_groups=False).reset_index()
    by_regime = trades.groupby(["strategy", "regime"]).apply(agg, include_groups=False).reset_index()
    return overall, by_regime[by_regime["trades"] >= _MIN_CELL_TRADES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=int, default=60, help="how many of tracking_universe.txt to use")
    parser.add_argument("--strategies", default="rsi_dip_recovery,macd,rsi,sweep_outside,donchian,ema_cross")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--source", choices=["yahoo", "tiingo"], default="yahoo")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="skip the index comparison (needs two extra fetches)")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    universe = __file__.rsplit("/", 1)[0] + "/tracking_universe.txt"
    with open(universe) as fh:
        tickers = [line.strip() for line in fh if line.strip()][: args.tickers]

    if args.source == "yahoo":
        from .cross_validate import fetch_yahoo_ohlcv
        fetch = lambda tk: fetch_yahoo_ohlcv(tk, args.start_date)
    else:
        from .data import fetch_ohlcv
        fetch = lambda tk: fetch_ohlcv(tk, start_date=args.start_date)

    exchanges = load_exchange_cache()
    index_cagr: dict[str, float] = {}
    if not args.skip_benchmark:
        for symbol in sorted({_DEFAULT_BENCHMARK, *_BENCHMARKS.values()}):
            series = fetch(symbol)["Close"]
            span = (series.index[-1] - series.index[0]).days / 365.25
            index_cagr[symbol] = compound_cagr([(series.iloc[-1] / series.iloc[0] - 1) * 100], span)

    per_ticker_cagr = []

    def one(ticker):
        df = fetch(ticker)
        if len(df) < 400:
            return None
        if not args.skip_benchmark:
            span = (df.index[-1] - df.index[0]).days / 365.25
            close = df["Close"]
            benchmark = benchmark_for(ticker, exchanges)
            rows = [{"strategy": "buy & hold the stock", "ticker": ticker, "benchmark": benchmark,
                     "cagr": compound_cagr([(close.iloc[-1] / close.iloc[0] - 1) * 100], span)}]
            for name in strategies:
                trades_ = backtest_strategy(df, name)
                if trades_:
                    rows.append({"strategy": name, "ticker": ticker, "benchmark": benchmark,
                                 "cagr": compound_cagr([t.return_pct for t in trades_], span)})
            per_ticker_cagr.extend(rows)
        return trade_excess(df, ticker, strategies)

    frames = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(one, tk) for tk in tickers]):
            try:
                result = future.result()
                if result is not None and len(result):
                    frames.append(result)
            except Exception:
                continue

    if not frames:
        print("no trades")
        return
    trades = pd.concat(frames, ignore_index=True)
    overall, by_regime = summarize(trades)

    print(f"\n{trades['ticker'].nunique()} tickers, {len(trades)} trades, from {args.start_date}")
    print("excess = trade return minus the mean return of holding the same number of bars")
    print("from any bar in the same regime. Positive means the entry timing added value.\n")
    header = f"{'strategy':>20} {'trades':>7} {'return %':>9} {'baseline %':>11} {'excess %':>9} {'t':>6} {'>0':>5}"
    print(header)
    for _, r in overall.sort_values("excess_pct", ascending=False).iterrows():
        print(f"{r['strategy']:>20} {int(r['trades']):>7} {r['mean_return_pct']:>9.2f} "
              f"{r['baseline_pct']:>11.2f} {r['excess_pct']:>9.2f} {r['t']:>6.2f} {r['pct_positive']:>4.0f}%")

    print(f"\n{'strategy':>20} {'regime':>16} {'trades':>7} {'excess %':>9} {'t':>6} {'>0':>5}")
    for _, r in by_regime.sort_values("excess_pct", ascending=False).iterrows():
        print(f"{r['strategy']:>20} {r['regime']:>16} {int(r['trades']):>7} "
              f"{r['excess_pct']:>9.2f} {r['t']:>6.2f} {r['pct_positive']:>4.0f}%")
    cells = len(by_regime)
    print(f"\n{cells} regime cells shown -- at p<0.05 roughly {cells * 0.05:.1f} would look significant")
    print("by chance, so read a single standout as a lead until it survives a correction.")

    if per_ticker_cagr:
        cagrs = pd.DataFrame(per_ticker_cagr)
        cagrs["index_cagr"] = cagrs["benchmark"].map(index_cagr)
        cagrs["vs_index"] = cagrs["cagr"] - cagrs["index_cagr"]
        print("\nAgainst the index each name could have been swapped for "
              f"({', '.join(f'{k} {v:.1f}% CAGR' for k, v in sorted(index_cagr.items()))}):\n")
        print(f"{'strategy':>22} {'med CAGR %':>11} {'vs index':>9} {'% beating index':>16}")
        ranked = sorted(
            ((g["vs_index"].median(), name, g["cagr"].median(), (g["vs_index"] > 0).mean() * 100)
             for name, g in cagrs.groupby("strategy")),
            reverse=True,
        )
        for gap, name, median, beat in ranked:
            note = "  <-- doing nothing" if name.startswith("buy & hold") else ""
            print(f"{name:>22} {median:>11.1f} {gap:>+9.1f} {beat:>15.0f}%{note}")


if __name__ == "__main__":
    main()
