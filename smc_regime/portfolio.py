"""One capital pool across the whole universe, instead of one per ticker.

WHY THIS EXISTS. Every return figure elsewhere in this project is
per-trade, or a per-ticker equity curve that compounds one name's trades
against each other. Neither is a portfolio. The per-ticker curves produce
drawdowns like rsi2's -97.6% in trending/up, which is not the risk of
trading the strategy -- it is the risk of putting the entire account into
one ticker, repeatedly, with no stop. A per-trade excess of +0.411% says
the entry timing carries information; it says nothing about what an
account running the strategy would have done.

WHAT A PORTFOLIO ADDS THAT PER-TICKER CURVES CANNOT SHOW.

  CONCURRENCY. A slot limit is the whole point. Connors' own research
  warns that RSI(2) signals cluster during selloffs, and they do here: on
  the worst days this rejects far more signals than it takes. Rejections
  are counted and reported, because a strategy whose signals mostly do not
  fit is a different proposition from one whose signals all do.

  CASH DRAG. Fewer signals than slots means idle capital. The per-trade
  mean never sees this; it averages only the bars where a trade existed.

  SELECTION, AND IT DOMINATES THE ANSWER. When signals outnumber slots
  something has to choose. On rsi2_prior_high across 415 names that is
  not a detail: at 10 slots, 88.9% of signals are rejected, so the
  selection rule decides most of the result.

  AN EARLIER VERSION DEFAULTED TO TICKER ORDER, on the reasoning that an
  alphabetical sort carries no information about which trade will work.
  That reasoning was wrong and the measurement showed it: at 10 slots and
  5bp per side, ticker order returned +17.59% CAGR while eight random
  seeds returned +17.72% to +28.44%, mean +23.11%. EVERY seed beat it.
  Alphabetical order is not neutral -- always preferring the earliest
  symbols concentrates the book into a fixed subset of the universe
  instead of sampling it, and that costs several points of CAGR.

  So the default is now random under a fixed seed: reproducible, but
  sampling the universe rather than one corner of it. One draw is still
  only one draw -- the seed spread (sd 3.8pp) is WIDER than the entire
  slippage sensitivity -- so a single simulate() call is not a usable
  estimate of anything. Use simulate_seeds(), which is what reports a
  mean and a spread.

  "ticker" remains available for a deterministic walk when debugging, and
  is documented as biased low rather than neutral. A rule like "take the
  lowest RSI" might well be better than either, but it would be an
  unvalidated second strategy smuggled into the measurement of the first.

  A REAL DRAWDOWN. Equity is marked to market every trading day on the
  open positions, so the drawdown is the account's, on the basis the
  index comparison uses. The per-trade-sequence drawdowns quoted elsewhere
  are not comparable to a buy-and-hold figure and should not be read
  against one.

NO LOOKAHEAD. Positions open at the entry bar's close and exit at the exit
bar's close, exactly as the trade list records them; nothing consults a
later bar. Slots free on the exit day BEFORE that day's entries are
considered, which is how it would actually work -- the alternative would
idle a slot for a day for no reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    equity: pd.Series          # daily mark-to-market account value
    taken: int                 # signals acted on
    rejected: int              # signals dropped for want of a slot
    trades: list               # (ticker, entry, exit, weight, return_pct)
    max_positions: int
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        s = self.stats
        return (f"slots={self.max_positions} taken={self.taken} rejected={self.rejected} "
                f"({self.rejection_rate*100:.1f}%) CAGR={s.get('cagr_pct', float('nan')):+.2f}% "
                f"maxDD={s.get('max_drawdown_pct', float('nan')):+.1f}% "
                f"invested={s.get('exposure_pct', float('nan')):.1f}%")

    @property
    def rejection_rate(self) -> float:
        total = self.taken + self.rejected
        return self.rejected / total if total else 0.0


def _stats(equity: pd.Series, invested_days: int) -> dict:
    if len(equity) < 2:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0]
    peak = equity.cummax()
    daily = equity.pct_change().dropna()
    return {
        "cagr_pct": (total ** (1 / years) - 1) * 100 if years > 0 else float("nan"),
        "total_return_pct": (total - 1) * 100,
        "max_drawdown_pct": float((equity / peak - 1).min() * 100),
        "exposure_pct": invested_days / len(equity) * 100,
        # Annualised, cash-included. Deliberately not called Sharpe: no
        # risk-free rate is subtracted, so it is return over volatility and
        # nothing more.
        "return_over_vol": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() else float("nan"),
        "years": years,
    }


def simulate(
    trades_by_ticker: dict[str, list],
    prices: dict[str, pd.DataFrame],
    max_positions: int = 10,
    capital: float = 100_000.0,
    slippage_pct: float = 0.0,
    selector: str = "random",
    seed: int | None = 0,
) -> PortfolioResult:
    """Run one capital pool over `trades_by_ticker`.

    Each position is opened at equity / max_positions, so the book scales
    with the account and a full book is fully invested. `slippage_pct` is
    charged per fill, matching run_backtest's convention, so a round trip
    pays it twice.

    `selector` decides who gets a slot when signals outnumber them.
    "random" under a fixed `seed` is the default: reproducible, and it
    samples the universe. "ticker" sorts by symbol, which is deterministic
    but measurably biased LOW -- see the module docstring. One seed is one
    draw and the spread across seeds is large, so prefer simulate_seeds()
    for any number you intend to quote. Trades already carrying a `size`
    below 1.0 -- from vol targeting -- keep it as a multiplier on the slot
    weight.
    """
    if selector not in ("ticker", "random"):
        raise ValueError(f"unknown selector {selector!r}")
    rng = np.random.default_rng(seed)

    # one shared calendar; a ticker absent on a date simply has no bar
    calendar = sorted(set().union(*(set(df.index) for df in prices.values())))
    calendar = pd.DatetimeIndex(calendar)

    opening: dict[pd.Timestamp, list] = {}
    for ticker, trades in trades_by_ticker.items():
        if ticker not in prices:
            continue
        for t in trades:
            opening.setdefault(t.entry_date, []).append((ticker, t))

    cash = capital
    held: dict[str, dict] = {}
    equity_curve, invested_days = [], 0
    taken = rejected = 0
    closed = []

    for date in calendar:
        # EXITS FIRST: a slot freed today is available today.
        for ticker in [t for t, p in held.items() if p["exit_date"] == date]:
            pos = held.pop(ticker)
            fill = pos["trade"].exit_price * (1 - slippage_pct / 100)
            cash += pos["shares"] * fill
            closed.append((ticker, pos["entry_date"], date, pos["weight"],
                           (fill / pos["entry_fill"] - 1) * 100))

        # ENTRIES: candidates are today's signals, capped by free slots.
        candidates = opening.get(date, [])
        if candidates:
            free = max_positions - len(held)
            ordered = sorted(candidates, key=lambda c: c[0])
            if selector == "random":
                ordered = [ordered[i] for i in rng.permutation(len(ordered))]
            for ticker, t in ordered:
                if ticker in held:
                    continue          # engine is single-position per ticker
                if free <= 0:
                    rejected += 1
                    continue
                slot = (cash + sum(p["shares"] * _price(prices, tk, date, p)
                                   for tk, p in held.items())) / max_positions
                weight = slot * min(getattr(t, "size", 1.0) or 1.0, 1.0)
                if weight <= 0 or weight > cash:
                    weight = min(weight, cash)
                if weight <= 0:
                    rejected += 1
                    continue
                fill = t.entry_price * (1 + slippage_pct / 100)
                shares = weight / fill
                cash -= shares * fill
                held[ticker] = {"shares": shares, "entry_fill": fill, "weight": weight,
                                "entry_date": date, "exit_date": t.exit_date, "trade": t}
                taken += 1
                free -= 1

        mark = sum(p["shares"] * _price(prices, tk, date, p) for tk, p in held.items())
        equity_curve.append(cash + mark)
        if held:
            invested_days += 1

    equity = pd.Series(equity_curve, index=calendar, name="equity")
    return PortfolioResult(equity=equity, taken=taken, rejected=rejected, trades=closed,
                           max_positions=max_positions,
                           stats=_stats(equity, invested_days))


@dataclass
class SeedSweep:
    """simulate() repeated over seeds, with the spread kept rather than hidden.

    A single run's CAGR is one draw from a distribution whose width is set
    by which signals happened to win a slot. On rsi2_prior_high at 10 slots
    that width (sd ~3.8pp) exceeds the whole effect of going from 0 to 10bp
    of slippage, so a number quoted without it is not meaningful. `mean`
    and `sd` are reported together for that reason, and `runs` is kept so a
    caller can look at any individual draw.
    """
    runs: list
    seeds: list

    def spread(self, key: str = "cagr_pct") -> dict:
        vals = [r.stats[key] for r in self.runs if key in r.stats]
        if not vals:
            return {}
        arr = np.asarray(vals, float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()), "max": float(arr.max()), "n": len(arr)}

    def __str__(self) -> str:
        c, d = self.spread("cagr_pct"), self.spread("max_drawdown_pct")
        rej = np.mean([r.rejection_rate for r in self.runs]) * 100
        return (f"{c['n']} seeds: CAGR {c['mean']:+.2f}% +/- {c['sd']:.2f} "
                f"(range {c['min']:+.2f} to {c['max']:+.2f})  "
                f"maxDD {d['mean']:+.1f}% +/- {d['sd']:.1f}  rejected {rej:.1f}%")


def simulate_seeds(
    trades_by_ticker: dict[str, list],
    prices: dict[str, pd.DataFrame],
    seeds: int | list[int] = 8,
    **kwargs,
) -> SeedSweep:
    """Run simulate() across several random tie-breaks and keep the spread.

    This is the entry point for any figure that will be quoted. `seeds` is
    a count (0..n-1) or an explicit list. `selector` is forced to "random"
    -- averaging over a deterministic rule would just repeat it n times.
    """
    kwargs.pop("selector", None)
    kwargs.pop("seed", None)
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    runs = [simulate(trades_by_ticker, prices, selector="random", seed=s, **kwargs)
            for s in seed_list]
    return SeedSweep(runs=runs, seeds=seed_list)


def _price(prices: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp, pos: dict) -> float:
    """Last known close at or before `date` -- a ticker can be missing a bar
    (halt, holiday mismatch) without the position vanishing from the book."""
    df = prices[ticker]
    try:
        return float(df.at[date, "Close"])
    except KeyError:
        prior = df.index[df.index <= date]
        if len(prior) == 0:
            return pos["entry_fill"]
        return float(df.at[prior[-1], "Close"])
