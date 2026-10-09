# StockMarketAnalysis

## SMC regime classifier

`smc_regime` classifies a stock's recent price action into one of three regimes:

- **choppy** -- no persistent direction, price churns back and forth
- **trending** -- steady, efficient move in one direction
- **parabolic** -- trending *and* accelerating (volatility expanding, price stretched
  far from its moving average, path curving rather than straight)

This is step one of a two-step workflow: classify the regime, then evaluate which
trading strategy performs best conditioned on that regime. Step two -- the
regime-conditioned backtest harness -- is implemented; see below.

### Usage

Requires a [Tiingo](https://www.tiingo.com/) API key (daily EOD data works on the
free tier; hourly/intraday needs a paid tier for real historical depth).
Set it as an environment variable -- never commit it:

```bash
export TIINGO_API_KEY=your_token_here
pip install -r requirements.txt
python -m smc_regime.cli AAPL --period 6mo
```

Or from Python:

```python
from smc_regime import fetch_ohlcv, classify_regime

df = fetch_ohlcv("AAPL", period="6mo")
result = classify_regime(df)
print(result[["regime", "direction"]].tail())
```

### How classification works

Rule-based, not ML -- deliberately, so the thresholds can be inspected and tuned
against real charts before anything more complex is layered on top.

| Signal | Metric | Role |
|---|---|---|
| Trendiness | Kaufman Efficiency Ratio, ADX | separates choppy from trending |
| Volatility acceleration | ATR% and its rate of change | flags expanding volatility |
| Extension | Distance from EMA in ATR units | flags overextended price |
| Curvature | R² gain of quadratic vs. linear fit | flags an accelerating (convex) path vs. a straight trend |

A bar is `parabolic` only if it's already `trending` AND stretched from its EMA AND
volatility is accelerating AND the price path is curving -- all four, not just one,
to avoid mislabeling an ordinary strong trend as parabolic.

All thresholds live in `smc_regime.regime.RegimeThresholds` and can be overridden:

```python
from smc_regime import classify_regime, RegimeThresholds

result = classify_regime(df, RegimeThresholds(er_trend_min=0.25, extension_parabolic_min=2.5))
```

### Regime-conditioned backtest

`smc_regime.backtest_cli` backtests every strategy in `smc_regime.strategies.STRATEGIES`
(RSI mean reversion, Bollinger mean reversion, MACD crossover, EMA 20/50 trend cross,
Donchian breakout, Supertrend, VWAP trend structure, plus three RSI+MACD combinations --
oversold-reversal, trend-continuation, and RSI-filtered MACD) across any list of tickers,
then tags every individual trade with the regime that was active *on its entry date* --
not the ticker's current regime -- and aggregates win rate / return by (regime, strategy).
This answers "which strategy performs best in which regime?" directly from trade-level
outcomes.

```bash
python -m smc_regime.backtest_cli AAPL MSFT TSLA --period 2y
```

```python
from smc_regime import fetch_ohlcv, collect_trades, summarize_by_regime

trades, latest_regime = collect_trades(["AAPL", "MSFT", "TSLA"], period="2y")
print(summarize_by_regime(trades))
```

All strategies are long-only, single-position, defined in `smc_regime.strategies.STRATEGIES`.

### Daily snapshot + recommendations

`smc_regime.daily_snapshot` runs the regime-conditioned backtest across a
tracking universe (`smc_regime/tracking_universe.txt`) at one or more
intervals, and stores trade-level results in a SQLite file
(`backtest_logs/smc_regime.db`) plus an append-only JSONL log
(`backtest_logs/regime_strategy_log.jsonl`). It also refreshes each
ticker's exchange and GICS sector into a `ticker_metadata` table.

```bash
python -m smc_regime.daily_snapshot --intervals 1d,1h
```

This runs daily via GitHub Actions (`.github/workflows/daily-snapshot.yml`),
which commits the JSONL log and ticker-metadata cache back to the repo
directly -- set `TIINGO_API_KEY` as a repository secret for it to run.

`backtest_logs/smc_regime.db` itself is **not** committed to git (it's
rewritten in full every night, which would grow the repo's history
unboundedly and risks hitting GitHub's 100MB per-file limit). It's
published as a GitHub Release asset instead. After cloning, fetch it with:

```bash
python -m smc_regime.fetch_db
```

`smc_regime.recommend_cli` classifies a ticker's current regime and looks up
the best-fitting strategy from the SQLite store, falling back from
symbol-specific stats (once a ticker has >= 15 of its own trades in that
regime/direction) to sector-level stats to the full-population pooled stats:

```bash
python -m smc_regime.recommend_cli AAPL
```

### Running a strategy as one account, in and out of sample

Every other measurement here is per-trade or per-ticker. Neither is
tradeable: a per-trade edge says nothing about how much of it a book with a
finite number of slots can actually capture, and a per-ticker equity curve
quietly assumes unlimited capital. `smc_regime.portfolio_cli` runs the
strategy as ONE account against the index it could have bought instead.

```bash
python -m smc_regime.portfolio_cli --strategy rsi2_prior_high
python -m smc_regime.portfolio_cli --strategy rsi2_prior_high --oos
```

`--oos` splits at 2024-01-01, selects a configuration on the in-sample
window ONLY, and then evaluates that single winner on the holdout with
nothing re-picked. Trades straddling the split are dropped from both
windows (`tests/test_oos_split.py`).

What it found, and it is the most important negative result in this repo:
RSI(2)'s per-trade edge is real and survives every test of its own kind
(+0.411% excess, t=8.87 ticker-clustered, holding ex-2020 and across 82% of
tickers, break-even at 40.8bp per side) -- and does NOT survive being run
as an account. Out of sample at 5bp per side it returned +12.04% CAGR
against SPY's +21.07%, with a deeper drawdown (-27.1% vs -18.76%): lower
return AND more risk, in a window it was not selected on. Three reasons,
all visible only at the portfolio level:

- **Rejection runs 64-89%.** The strategy fires far more signals than a
  book can hold, so the per-trade statistics describe a population only a
  fraction of which is actionable.
- **The book ends up 99.9% invested.** On SPY alone RSI(2) earned 83% of
  the return with 28% exposure. Across 415 names there is always a signal,
  so that advantage disappears and what remains is a long equity portfolio
  with a thin entry overlay -- carrying the same market risk as the index.
- **Diversification is the one unambiguous win.** 10 -> 40 slots cut
  drawdown -38.8% -> -33.7% and cut seed dispersion +/-4.07 -> +/-0.80 for
  5pp of mean CAGR. Below ~20 slots the answer is dominated by an arbitrary
  tie-break rather than by the strategy, which is why every quoted figure
  is averaged over 8 random tie-breaks with its spread shown.

The module docstring carries the full result tables and the selection
protocol, including the pre-registered fallback rule that fired (all six
candidates failed the drawdown constraint) and why it was honoured as
written rather than rewritten once the candidates were visible.

### Pine Script: viewing the live regime on a TradingView chart

`pinescript/smc_regime_classifier.pine` is a hand-ported mirror of
`smc_regime.regime.classify_regime()` -- same Kaufman Efficiency Ratio / ADX
trend gate, same EMA-extension + ATR%-acceleration parabolic gate, same
rolling-slope direction call, with matching default thresholds. Add it as an
indicator on any TradingView chart to see the current regime (choppy /
trending / parabolic, with direction) shaded on the price panel, a live
stats table, and alerts on regime/direction changes -- no live data feed
from this repo required, since it computes everything from the chart's own
bars.

There's no automatic sync between the two: if `RegimeThresholds` or the
indicator formulas in `smc_regime/indicators.py` change, update the Pine
script by hand to match.

### Dashboards

The two HTML dashboards are versioned as **templates** -- markup carrying an
empty payload -- under `smc_regime/dashboards/`. A publishable page is built
by baking a fresh export into one:

```bash
# export and bake in one step
python -m smc_regime.build_dashboard setup_grades --out setup_grades.html
python -m smc_regime.build_dashboard regime_desk  --out regime_desk.html

# or bake a payload you already exported
python -m smc_regime.export_setup_score_data --out-file payload.json
python -m smc_regime.build_dashboard setup_grades --data payload.json --out page.html
```

The split is deliberate. A template is ~60KB of reviewable markup; a built
page is ~800KB, nearly all of it a snapshot of the database, which would
land as an unreadable diff every night. So the markup is tracked and the
built pages are not -- do not commit them.

Each template is a working page on its own: its payload is empty but
structurally complete, so opening one directly gives the real chrome in its
empty state. That makes the markup editable with no database around, and
means a broken template fails when you open it rather than during a publish.

### Next steps

- Validate labels by eye against a chart for a few tickers with known regimes, tune thresholds
- Add SMC structure signals (BOS/CHoCH frequency, order block respect rate) as a confirming layer
- Add commission/slippage modeling and walk-forward validation to the backtest harness
