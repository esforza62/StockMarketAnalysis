"""Run a strategy as ONE ACCOUNT, and split it in and out of sample.

Everything else in this project measures per-trade returns or a per-ticker
equity curve. Neither is tradeable. This asks the only question that
decides whether a measured edge is useful: what would an account running
it have done, against the index it could have bought instead.

    python -m smc_regime.portfolio_cli --strategy rsi2_prior_high
    python -m smc_regime.portfolio_cli --strategy rsi2_prior_high --oos

WHAT IT FOUND, 415 tickers of daily bars from 2019-01, costs charged per
fill so a round trip pays twice. Tables below were produced on bars through
2026-10-08; re-running on later bars moves every figure slightly, which is
a longer window rather than a discrepancy.

FULL PERIOD, seed-averaged over 8 tie-breaks. SPY buy-and-hold over the
same window: +17.37% CAGR, -33.72% drawdown.

    slots  bp/side   CAGR% (8 seeds)        maxDD%   rejected  vs SPY
       10        0   +29.91 +/- 4.29        -37.8      88.8%   yes
       10        5   +23.11 +/- 4.07        -38.8      88.8%   yes
       10       10   +16.67 +/- 3.85        -40.4      88.8%   mixed
       20        5   +18.92 +/- 1.25        -38.3      79.2%   mixed
       40        0   +23.34 +/- 0.84        -33.6      63.6%   yes
       40        5   +18.08 +/- 0.80        -33.7      63.6%   mixed
       40       10   +13.05 +/- 0.77        -33.9      63.6%   no

Three things in that table matter more than the headline CAGR.

  REJECTION RUNS 64-89%. At universe scale the strategy fires far more
  signals than a book can hold, so the per-trade statistics describe a
  population only a tenth of which can be acted on.

  THE BOOK IS 99.9% INVESTED. On SPY alone, RSI(2)'s appeal was 83% of the
  return for 28% exposure -- in cash four days in five. Across 415 names
  there is always a signal, so that advantage does not survive scaling.
  What remains is a long equity portfolio with a thin entry overlay.

  DIVERSIFICATION IS THE ONE UNAMBIGUOUS WIN. 10 -> 40 slots cut drawdown
  -38.8% -> -33.7% AND cut seed dispersion +/-4.07 -> +/-0.80, for 5pp of
  mean CAGR. At 10 slots the answer is dominated by an arbitrary
  tie-break; at 40 it is dominated by the strategy.

OUT OF SAMPLE, and this is the finding that matters. Protocol fixed before
any result was read: split 2024-01-01; select on IS only at 5bp over 8
seeds from {rsi2, rsi2_prior_high} x {10,20,40} slots, highest CAGR subject
to drawdown no worse than SPY's own; evaluate that one winner on OOS.
Trades straddling the boundary are dropped from both windows.

ALL SIX CANDIDATES FAILED THE DRAWDOWN CONSTRAINT. In the window that
contains the COVID crash every configuration drew down worse than SPY
(-33.73% to -38.6% against -33.70%); the 40-slot case missed by 0.03pp.
The full-period appearance of "index-matching drawdown" came partly from
including the gentle 2024-2026 stretch. The pre-registered fallback --
highest CAGR -- then picked rsi2_prior_high at 10 slots, which is also the
worst drawdown and widest spread in the table. A bad rule, honoured as
written, because rewriting it once the pick is visible is the failure the
split exists to detect.

    OOS, rsi2_prior_high @ 10 slots.  SPY OOS: +21.07% CAGR, -18.76% DD

    bp/side   OOS CAGR% (8 seeds)       OOS maxDD%   vs SPY
          0   +18.32 +/- 3.82           -25.8        mixed
          5   +12.04 +/- 3.59           -27.1        no
         10    +6.09 +/- 3.38           -28.7        no

In sample the same configuration returned +28.14% at 5bp. Out of sample it
returned +12.04% -- 9pp BEHIND the index, with -27.1% drawdown against
SPY's -18.76%. Lower return and more risk, in a window it was not chosen
on. Frictionless it does not clear the index either.

The per-trade edge is real and survives every test of its own kind:
+0.411% excess, t=8.87 ticker-clustered, holding ex-2020 and across 82% of
tickers, break-even at 40.8bp per side. It does not survive being run as
an account against the index in a window it was not selected on. Those are
consistent: against CASH the overlay pays, against SPY it pays roughly its
own transaction costs, because the book carries the same market risk
either way.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .backtest import backtest_strategy
from .portfolio import simulate_seeds

#: Fixed before any result was read -- see the module docstring.
OOS_SPLIT = "2024-01-01"
_CANDIDATE_SLOTS = (10, 20, 40)
_CANDIDATE_STRATEGIES = ("rsi2", "rsi2_prior_high")


def _cagr_dd(close: pd.Series) -> tuple[float, float]:
    years = (close.index[-1] - close.index[0]).days / 365.25
    cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100
    return cagr, float(((close / close.cummax()) - 1).min() * 100)


def split_trades(trades: list, split: pd.Timestamp) -> tuple[list, list, int]:
    """(in-sample, out-of-sample, dropped).

    A trade entering before the split and exiting after it is DROPPED from
    both windows rather than assigned to one: keeping it in-sample would
    put post-split prices there, and keeping it out-of-sample would credit
    the later window with an entry it did not make. Holds are a few bars,
    so the loss is small and the windows stay clean.
    """
    before, after, dropped = [], [], 0
    for t in trades:
        if t.entry_date < split and t.exit_date < split:
            before.append(t)
        elif t.entry_date >= split:
            after.append(t)
        else:
            dropped += 1
    return before, after, dropped


def sweep(trades_by_ticker, prices, benchmark, slots=_CANDIDATE_SLOTS,
          costs=(0.0, 0.05, 0.10), seeds=8, capital=100_000.0) -> pd.DataFrame:
    """Seed-averaged CAGR/drawdown for each (slots, cost) pair.

    `vs_benchmark` is keyed to the WORST seed, not the mean: "the worst
    tie-break still beat the index" is a claim worth making, "the average
    one did" is not, when the spread reaches 4pp.
    """
    bench_cagr, _ = _cagr_dd(benchmark)
    rows = []
    for n in slots:
        for bp in costs:
            sw = simulate_seeds(trades_by_ticker, prices, seeds=seeds,
                                max_positions=n, capital=capital, slippage_pct=bp)
            c, d = sw.spread("cagr_pct"), sw.spread("max_drawdown_pct")
            rows.append({
                "slots": n, "bp_per_side": bp * 100,
                "cagr_mean": c["mean"], "cagr_sd": c["sd"],
                "cagr_min": c["min"], "cagr_max": c["max"],
                "maxdd_mean": d["mean"],
                "rejected_pct": float(np.mean([r.rejection_rate for r in sw.runs]) * 100),
                "vs_benchmark": ("yes" if c["min"] > bench_cagr
                                 else "mixed" if c["max"] > bench_cagr else "no"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="rsi2_prior_high")
    p.add_argument("--tickers", type=int, default=500)
    p.add_argument("--start-date", default="2019-01-01")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--oos", action="store_true",
                   help=f"split at {OOS_SPLIT}, select on in-sample only, report both windows")
    p.add_argument("--split", default=OOS_SPLIT)
    args = p.parse_args()

    from .cross_validate import fetch_yahoo_ohlcv
    universe = __file__.rsplit("/", 1)[0] + "/tracking_universe.txt"
    with open(universe) as fh:
        tickers = [ln.strip() for ln in fh if ln.strip()][: args.tickers]
    if args.benchmark not in tickers:
        tickers.append(args.benchmark)

    prices, trades = {}, {}
    for tk in tickers:
        try:
            df = fetch_yahoo_ohlcv(tk, args.start_date)
            if df.empty:
                continue
            prices[tk] = df
            trades[tk] = backtest_strategy(df, args.strategy, interval="1d")
        except Exception:
            continue
    bench = prices[args.benchmark]["Close"]
    print(f"{len(prices)} tickers, {sum(map(len, trades.values()))} trades, "
          f"{args.strategy}, from {args.start_date}\n")

    if not args.oos:
        c, d = _cagr_dd(bench)
        print(f"{args.benchmark} buy & hold: CAGR {c:+.2f}%  maxDD {d:.1f}%\n")
        print(sweep(trades, prices, bench, seeds=args.seeds,
                    capital=args.capital).to_string(index=False))
        return

    split = pd.Timestamp(args.split, tz=bench.index.tz)
    is_bench, oos_bench = bench[bench.index < split], bench[bench.index >= split]
    ic, idd = _cagr_dd(is_bench)
    oc, odd = _cagr_dd(oos_bench)
    print(f"IS  to {split.date()}  {args.benchmark} CAGR {ic:+.2f}%  maxDD {idd:.2f}%")
    print(f"OOS from {split.date()}  {args.benchmark} CAGR {oc:+.2f}%  maxDD {odd:.2f}%\n")

    prices_is = {t: d[d.index < split] for t, d in prices.items()}
    prices_oos = {t: d[d.index >= split] for t, d in prices.items()}

    # Build each candidate's split trades once, then score every (strategy,
    # slots) pair on the IN-SAMPLE window only.
    dropped_total = 0
    oos_trades: dict[str, dict] = {}
    scored: list[tuple[float, float, str, int]] = []
    for strat in _CANDIDATE_STRATEGIES:
        tr_is, tr_oos = {}, {}
        for tk, df in prices.items():
            try:
                a, b, drop = split_trades(
                    backtest_strategy(df, strat, interval="1d"), split)
            except Exception:
                continue
            tr_is[tk], tr_oos[tk] = a, b
            dropped_total += drop
        oos_trades[strat] = tr_oos
        for n in _CANDIDATE_SLOTS:
            sw = simulate_seeds(tr_is, prices_is, seeds=args.seeds, max_positions=n,
                                capital=args.capital, slippage_pct=0.05)
            c, d = sw.spread("cagr_pct"), sw.spread("max_drawdown_pct")
            scored.append((c["mean"], d["mean"], strat, n))
            print(f"  {strat:16s} {n:3d} slots  CAGR {c['mean']:+7.2f} +/-{c['sd']:4.2f}  "
                  f"maxDD {d['mean']:+7.2f}  "
                  f"{'clears' if d['mean'] >= idd else 'fails'} the DD constraint")

    # The pre-registered rule: highest IS CAGR among those whose drawdown is no
    # worse than the benchmark's. The fallback when NONE clears is highest CAGR
    # outright -- a badly designed tie-break, since it selects on the one axis
    # the constraint was there to bound, so it tends to pick the widest-spread,
    # worst-drawdown cell. Applied as written anyway: rewriting the rule once
    # the candidates are visible is exactly the failure a holdout detects.
    clearing = [s for s in scored if s[1] >= idd]
    if clearing:
        _, _, strat, n = max(clearing, key=lambda s: s[0])
        basis = "cleared the drawdown constraint"
    else:
        _, _, strat, n = max(scored, key=lambda s: s[0])
        basis = "NO candidate cleared the drawdown constraint -- fallback: highest IS CAGR"
    tr_oos = oos_trades[strat]

    print(f"\nSELECTED {strat} at {n} slots -- {basis}.")
    print(f"Boundary trades dropped from both windows: {dropped_total}\n")
    print(f"OUT OF SAMPLE ({strat} @ {n} slots, nothing re-picked):")
    print(sweep(tr_oos, prices_oos, oos_bench, slots=(n,), seeds=args.seeds,
                capital=args.capital).to_string(index=False))


if __name__ == "__main__":
    main()
