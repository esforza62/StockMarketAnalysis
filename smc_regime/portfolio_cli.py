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

WALK-FORWARD, five sequential folds (OOS 2022-2026), which is where the
single split above turns out to have been misleading in BOTH directions.

The pre-registered rule from the single split, applied per fold, picked
10 slots by FALLBACK in 5 of 5 folds -- no candidate's drawdown ever
cleared the benchmark's, in any window. It beat SPY in 2/5 folds at 0bp,
1/5 at 5bp, 0/5 at 10bp, mean gap -5.52pp, with drawdown worse than SPY
in 5 of 5. So the rule is not a tie-break that occasionally misfires: it
reliably selects the worst cell in the grid, because maximising in-sample
CAGR always reaches for concentration.

It also refutes the reason the walk-forward was run. The hypothesis was
that mean reversion should fare BETTER in weak or choppy markets and worse
in persistent trends, which would have made the single split's failure an
artefact of testing on 2024-2026. The correlation of excess against share
of days SPY closed above its 200-day average is +0.71 -- the opposite
sign. 2022, the only genuinely bad year (19% of days above the 200sma),
is the WORST fold at -11.5pp. n=5 and that correlation leans heavily on
one fold, so it is a direction and not a measurement, but the direction
is backwards from the premise.

FIXED CONFIGURATION, no selection step at all, so nothing to overfit:

    slots   beat SPY   mean gap   mean maxDD   SPY maxDD
       10      4/10     -1.58pp       -20.0%      -14.1%
       20      4/10     -1.45pp       -16.3%      -14.1%
       40      4/10     +0.45pp       -14.9%      -14.1%

Read the convergence, not the ranking. As the book widens 10 -> 40 the gap
to SPY goes to zero AND the drawdown goes to SPY's. That is the clearest
statement of the whole exercise: 40 equal-weighted names drawn from 415
large caps, held essentially always, IS the index with tracking error,
whichever rule picked the names. The strategy's contribution is which 40
and when, and holding 40-of-415 continuously dilutes that to nothing.

RANKED SELECTION AND A QUALITY FLOOR (see portfolio.simulate). Scores are
RSI(2) depth at the entry bar, so the deepest signals get the scarce
slots, and a floor leaves a slot empty rather than filling it with a
shallow signal. Effect against the random baseline, 20 cells each, 5bp:

    mode      mean   median          range   helps   fill    maxDD
    ranked   +1.57    +2.10   -6.7 .. +16.5   13/20    86%   -15.0%
    floor3   +0.47    -1.48  -15.5 .. +25.7    9/20    74%   -14.5%
    floor5   +1.45    +1.48  -27.3 .. +34.4   12/20    57%   -11.7%
    floor7   -7.83    -5.41  -41.9 .. +22.9    6/20    23%    -6.6%
    floor8  -11.88   -11.54  -40.5 .. +17.4    3/20     7%    -2.9%

THE FLOOR IS AN EXPOSURE DIAL, NOT AN EDGE. floor5's +1.45 mean hides
+23.1 / -22.7 / -5.3 / -1.3 / +13.5 across the five folds: it wins where
the market fell and loses where it rose, in proportion to the cash it
holds. That is a short bet with no timing in it, and quoting the mean
would be the most misleading number available. Its best single cell
(+31.34% in 2022 against SPY's -18.84%) is the same coin landing well.

RANKING IS THE ONE THING THAT HELD: +1.57pp, positive in ALL FIVE folds
(+2.90 +0.18 +0.24 +2.74 +1.79), at unchanged exposure, so it is a pure
selection effect rather than a market bet. It is also small, two of those
folds are indistinguishable from zero, and +1.57pp does not close a
5.52pp gap. RSI(2) depth carries a little information about which signal
pays, and not much.

THE DECISIVE NUMBER: of the 48 cells in 2024 and 2025 -- the two clean
trending folds -- ZERO beat SPY. Across all 120 cells, 39 beat it at 5bp
and 31 at 10bp, concentrated in 2022 where SPY fell 19% and nearly any
less-invested book beat it by not falling.

WHAT DID SURVIVE, for the fourth time in this project: risk reduction,
never excess return. floor5 produced an indistinguishable CAGR on 56% of
the capital at risk with -11.7% drawdown, beating SPY's drawdown in 14 of
20 cells against random's 4 of 20; floor7 beat it in 20 of 20. Stops did
this (a 2% stop on ema_cross cut tickers past -50% from 20.7% to 0.2% for
0.054pp of excess) and diversification did this (10 -> 40 slots, -38.8%
-> -33.7%). Three independent mechanisms, one shape of answer: this
universe and this signal family can be made meaningfully less risky and
cannot be made to out-return the index.

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
