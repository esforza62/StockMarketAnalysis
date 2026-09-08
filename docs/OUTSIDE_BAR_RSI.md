# Outside Bar Finder (wick filtered + RSI) — port and backtest

Source: `pinescript/outside_bar_rsi.pine`, the user's TradingView indicator.
Python port: `outside_bar_reversal` / `outside_bar_rsi` in
`smc_regime/strategies.py`, registered as `outside_bar` and `outside_bar_rsi`.

## What the pattern is

An outside bar engulfs the prior bar's entire range (higher high *and* lower
low). The wick filter is what turns that into a reversal read rather than a
plain volatility expansion: a bullish signal needs its **lower** wick to be at
least `wickRatio` times its upper wick (sellers pushed price down and were
rejected) and must follow a **down** bar; the bearish signal is the mirror.
RSI(21) adds a second tier — the indicator highlights bullish signals fired
while RSI is still *below* its midline, and bearish ones fired while RSI is
still *above* it.

## What had to be decided rather than ported

Per the porting rule in `ROADMAP.md`, these are flagged, not silently
approximated:

- **The Pine script is an indicator, not a strategy.** It has no position, no
  exit and no stop. The bearish signal — a short entry in the original — is
  used here as the long-only strategy's **exit**. That is the closest faithful
  reading available in a single-position long-only engine, but it is a choice.
- **The short side is dropped entirely**, as the engine cannot model it. The
  event study below says this matters: the short side is where the edge is.
- **The exit has no guarantee of firing.** Wick-filtered outside bars are rare
  (~2–4 bullish signals per ticker per year on daily bars), so 47% of the
  `outside_bar` entries in this run were still open at the end of the data and
  never became trades at all.

## How this run was measured

- 411 of the 415 tickers in `smc_regime/tracking_universe.txt`, daily bars,
  2021-09-09 → 2026-09-08 (~1,250 bars each).
- Regime tagging, trade construction and aggregation all use the repo's own
  code (`confirmed_regime(confirm_bars=3)`, `run_backtest`,
  `summarize_by_regime`) — same treatment every other strategy gets.
- **Data caveat:** `TIINGO_API_KEY` was not available in the session this ran
  in, so bars came from Yahoo's chart endpoint instead, split/dividend-adjusted
  the same way `data.py` adjusts Tiingo (OHLC scaled by the adjclose/close
  ratio). Prices spot-checked against the daily aggregates from the Massive
  MCP server and matched. Re-running with the real fetcher should shift the
  third decimal, not the conclusion.
- The universe is today's tracking list, so it carries survivorship bias — but
  the drift control below compares every trade against the *same ticker's* own
  average return, which is exactly the comparison survivorship does not
  distort.

## Headline results (all regimes pooled)

| strategy | trades | win rate | avg return/trade | avg hold | worst ticker drawdown |
|---|---|---|---|---|---|
| `rsi_dip_recovery` (baseline) | 1,694 | 78.3% | +11.07% | 242d | −98.4% |
| `rsi_dip_trend_filter` (baseline) | 517 | 78.7% | +9.81% | 250d | −59.9% |
| `rsi` (baseline) | 2,027 | 76.4% | +7.48% | 143d | −97.8% |
| **`outside_bar` (wick ratio 2.0)** | 1,271 | 53.0% | +4.14% | 119d | −88.2% |
| **`outside_bar`** | 3,797 | 52.7% | +3.72% | 82d | −98.3% |
| **`outside_bar_rsi`** | 1,961 | 52.3% | +3.21% | 86d | −96.8% |
| `donchian` (baseline) | 6,038 | 40.9% | +2.95% | 64d | −98.5% |
| `supertrend` (baseline) | 7,552 | 41.0% | +2.25% | 48d | −99.4% |
| **`outside_bar_rsi` + 8% stop** | 2,390 | 37.1% | +1.21% | 43d | −52.8% |
| `macd` (baseline) | 20,317 | 37.1% | +0.90% | 18d | −96.8% |
| **`outside_bar_rsi` + 20-bar max hold** | 2,490 | 50.0% | +0.84% | 25d | −71.9% |

Read on its own that looks respectable — mid-table, ahead of `macd`,
`supertrend` and `donchian`, behind the RSI family. Three things say otherwise.

### 1. The RSI filter the indicator highlights costs money

`outside_bar_rsi` (RSI(21) < 50 at entry) is *worse* than the unfiltered
signal on every measure — +3.21% vs +3.72% per trade, 52.3% vs 52.7% win rate
— on roughly half the trades. The colour distinction the indicator draws is
real, but it does not select the better signals.

Raising the wick ratio to 2.0 is the one parameter change that helped
(+4.14%/trade), at the cost of two thirds of the signals.

### 2. Both risk controls destroy the edge

An 8% stop and a 20-bar max hold each cut average return per trade by roughly
three quarters. This is the same pattern the repo already documented for
`rsi_dip_recovery`: capping the loser just books it and re-enters. The stop
does halve worst-case drawdown (−96.8% → −52.8%), so it is a real risk/return
trade, not a free lunch — but there is very little return left to protect.

### 3. Almost all of the return is market drift, not signal

An 82-day long-only hold in 2023–2026 makes money without any signal at all.
Comparing each trade against the mean of *every* same-length holding window on
the same ticker over the same five years:

| strategy | avg return | random timing, same holds | excess | t |
|---|---|---|---|---|
| `rsi` | +7.48% | +6.14% | **+1.34%** | +2.00 |
| `macd` | +0.90% | +0.91% | −0.00% | −0.03 |
| `outside_bar` | +3.72% | +4.39% | **−0.67%** | −1.39 |
| `outside_bar_rsi` | +3.21% | +3.98% | **−0.76%** | −1.07 |
| `outside_bar` (wick 2.0) | +4.14% | +5.63% | **−1.50%** | −1.85 |
| `rsi_dip_recovery` | +11.07% | +12.67% | −1.60% | −1.59 |
| `rsi_dip_trend_filter` | +9.81% | +12.25% | −2.44% | −1.39 |

Every outside-bar variant underperforms random entry timing on the same
ticker for the same holding period. So does `rsi_dip_recovery`, the repo's
current headline strategy — worth its own follow-up, since its +11%/trade has
been read as edge and is mostly drift over a very long average hold.

The year split says the same thing from another angle: `outside_bar` averaged
−1.94% (2021) and −1.79% (2022) per trade, then +7.32%, +7.43%, +5.67% and
+3.37% in 2023–2026. It makes money when the market goes up.

## Event study: what the pattern actually predicts

The exit rule above is mine, not the indicator's, so it could be what is
losing the money. Testing the raw signal with no strategy attached — forward
return after each signal minus that ticker's average N-bar return, pooled over
all 411 tickers:

| cohort | horizon | excess return | t |
|---|---|---|---|
| bull outside bar | 5 bars | −0.30% | −3.98 |
| bull outside bar | 10 bars | −0.35% | −3.38 |
| bull + RSI < 50 | 5 bars | −0.33% | −2.52 |
| bull + RSI > 50 | 5 bars | −0.29% | −3.07 |
| bear outside bar | 3 bars | +0.10% | +1.89 |
| bear outside bar | 10 bars | +0.19% | +1.91 |
| **bear outside bar** | **60 bars** | **−0.86%** | **−3.23** |
| **bear + RSI > 50** | **20 bars** | **−0.83%** | **−4.47** |
| **bear + RSI > 50** | **60 bars** | **−2.08%** | **−6.08** |

The bullish signal is followed by *below*-average returns over the next 1–2
weeks, consistently and significantly — the opposite of what it claims. Over
the same horizon the bearish signal is followed by slightly *above*-average
returns, also the opposite of its claim: both are short-horizon mean reversion
after an outsized bar.

The one genuinely strong result in the whole study is the bearish signal at
long horizons, and specifically the indicator's own highlighted bear tier:
**a bearish wick-filtered outside bar with RSI(21) above 50 precedes 2.08%
underperformance over the next 60 bars, t = −6.1.** That is the largest and
most significant effect measured here, it is on the side the Pine script
already colours specially, and it is exactly the side this long-only engine
cannot trade.

## Where this leaves it

Both variants are registered in `STRATEGIES`, so the nightly snapshot will
keep measuring them alongside everything else and the SQLite store will
accumulate live evidence either way. But on this evidence the long side is not
worth trading, and the RSI-below-midline filter that the indicator highlights
makes it worse rather than better.

The result worth acting on is the short side, which needs an engine change to
test properly (`run_backtest` is single-position long-only, and shorting is on
the "don't approximate, flag it" list in `ROADMAP.md`). A short-side harness
would be a real addition, not a tweak — that is a decision for the user, not
something to bolt on quietly.
