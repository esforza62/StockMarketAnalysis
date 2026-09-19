---
name: brief
description: Produce an investor-style briefing for a ticker -- recent price action, trend/volatility levels, regime classification, news sentiment, and a short takeaway. Use when asked to "brief", "review", or "summarize recent movement" for a stock (e.g. "/brief TSLA", "brief me on NVDA", "what's going on with AAPL"). Sources data from Tiingo via this repo's smc_regime modules.
---

# Ticker briefing

Produce a short investor-style briefing for one ticker. Default lookback is 1
year of daily bars; the briefing itself focuses on the last ~3 months.

## Prerequisites

`TIINGO_API_KEY` must be set in the environment. Every fetch below reads it.
If it is missing, stop and say so -- do not silently fall back to another
data source, since mixing providers is how adjusted/unadjusted figures get
crossed (see "Adjusted closes" below).

Run from the repo root with `PYTHONPATH=.` so `smc_regime` imports.

## Gathering the data

Prefer this repo's existing modules over raw HTTP -- they handle chunking,
adjustment and caching already:

```python
from smc_regime import fetch_ohlcv, classify_regime
from smc_regime.technicals import technical_snapshot
from smc_regime.news import ticker_sentiment, fetch_news
from smc_regime.valuation import fetch_valuation
from smc_regime.regime import confirmed_regime, regime_streak_bars

df = fetch_ohlcv("TSLA", period="1y")        # daily, adjusted
snap = technical_snapshot(df)                 # MAs, returns, vol
regime = classify_regime(df)                  # choppy / trending / parabolic
sent = ticker_sentiment("TSLA", days=7)       # VADER over Tiingo headlines
val = fetch_valuation("TSLA")                 # P/E, next earnings date
```

`fetch_valuation` is a separate (non-Tiingo) source and may return `None` --
treat missing valuation as missing, do not substitute a guess.

Tiingo's `/tiingo/fundamentals` endpoint is **DOW-30 only** on Free and Power
plans and returns HTTP 400 for everything else. Do not present its error as a
data gap in the company; say the plan does not cover it.

## What the briefing must contain

1. **Where it closed** -- last close, % change, volume vs the 60-day average,
   day's range, and where in that range it closed.
2. **Performance table** -- 1D / 1W / 1M / 3M / 6M / 1Y returns, plus a
   benchmark comparison (SPY and QQQ over the same 3M window) so
   stock-specific moves are separated from market moves.
3. **Levels** -- 52-week high/low with distance from each, 50-day and 200-day
   MAs and which side price is on, realized vol at 1M vs 3M.
4. **Regime** -- the `classify_regime` label, and how many bars it has held
   (`regime_streak_bars`). This is the repo's own signal; lead with it rather
   than re-deriving trend by eye.
5. **The quarter's defining moves** -- the 4-5 largest close-to-close moves in
   the last 63 sessions, with volume. A move on 3x volume means something
   different from the same move on average volume; say which.
6. **Context** -- headlines from `fetch_news`, with the sentiment score. Tie
   each item to the thesis, not just the date.
7. **Takeaway** -- 2-3 sentences. Name the next catalyst and what would
   change the read.

## Rules

**Adjusted closes.** Use close-to-close on adjusted prices for all return
figures. Open-to-close understates gap moves badly -- an earnings drop that is
-14% close-to-close can read as -6% open-to-close, which is the difference
between "the defining event of the quarter" and "a bad day". `fetch_ohlcv`
already returns adjusted data; do not mix it with an unadjusted series.

**Separate stock from market.** A -9% quarter means little without knowing
the index did +2%. Always give the relative number.

**Do not give investment advice.** Describe what the data shows and what the
risks are. Close with a one-line disclaimer. Never recommend buying, selling
or holding, and never state a price target.

**Flag stale data.** If the last bar is not the previous trading day, say so
up front -- on a weekend or holiday the "current" price is the last close.

**Say what failed.** If a fetch errors or a plan tier blocks an endpoint,
state it in the briefing rather than quietly omitting that section.
