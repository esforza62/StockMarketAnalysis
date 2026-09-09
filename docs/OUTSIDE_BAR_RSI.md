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
all 411 tickers. **The t-stats in this table are pooled and therefore
optimistic** — overlapping windows on the same ticker, and 411 tickers moving
together on the same days, are not independent observations. The follow-up
section re-runs them clustered; read this table for effect sizes and the
clustered one for significance:

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

The largest effect in the study is the bearish signal at long horizons, and
specifically the indicator's own highlighted bear tier: a bearish
wick-filtered outside bar with RSI(21) above its midline precedes ~2%
underperformance over the next 60 bars. It is on the side the Pine script
already colours specially, and exactly the side this long-only engine cannot
trade. How much of it survives clustering — and which midline threshold holds
up — is the next section.

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


# Follow-up: tighter RSI bands, and RSI divergence

Two questions from the user after the first pass: does tightening the RSI
confirmation to **below 45 / above 55** sharpen the signal, and does adding
**RSI divergence** help? Same universe, same period, same method.

## Clustering first, because it changes the answers

The t-stats in the first event-study table pool every signal as an
independent observation. They are not: 60-bar windows on one ticker overlap
each other, and all 411 tickers move together on the same days. Recomputing
the key cohorts three ways — pooled, clustered by ticker (411 units), and
clustered by calendar month (61 units, which is what absorbs the market-wide
component):

| cohort | horizon | excess | t pooled | t by ticker | t by month | months positive |
|---|---|---|---|---|---|---|
| bull outside bar | 5 bars | −0.30% | −3.98 | −3.63 | −1.69 | 47% |
| bull + RSI < 45 | 5 bars | −0.38% | −2.10 | −2.19 | +0.31 | 47% |
| bear + RSI > 50 | 60 bars | −2.08% | −6.08 | −5.27 | −1.87 | 41% |
| **bear + RSI > 55** | **60 bars** | **−2.34%** | **−5.80** | **−4.73** | **−2.85** | **34%** |
| **bull divergence, no outside bar** | **5 bars** | **+0.24%** | **+3.81** | **+3.76** | **+2.73** | **66%** |
| bull divergence, no outside bar | 20 bars | +0.65% | +4.99 | +5.01 | +2.27 | 69% |

Two corrections to the first pass fall out of this. The bullish outside bar's
short-horizon underperformance is consistent (negative pooled, by ticker, and
by month) but only weakly significant once clustered — it is best read as "no
edge", not as a reliable fade. And the headline bear result at RSI > 50 does
not survive month clustering (t = −1.87).

## Tighter bands: yes on the short side, no on the long side

**Bear > 55 is a real improvement.** It is the one cohort in the study
significant under *both* clustering schemes: −2.34% over 60 bars, month-
clustered t = −2.85 against −1.87 for the > 50 version, underperforming in
two thirds of calendar months. It costs 38% of the signals to get there
(2,545 vs 4,107). Tightening further reverses it — > 60 and > 65 give
−1.62% and −1.88% on progressively weaker t-stats.

**Bull < 45 does nothing.** Pooled it looks negative (t = −2.10 at 5 bars),
month-clustered it is indistinguishable from zero (+0.31), and as a traded
strategy it is worse than the < 50 version: −1.10% vs −0.76% against random
timing. Bands below 40 have too few signals to read — the one cell that looks
good (< 35 at 20 bars, +1.65%, t = 2.07) is one cell out of 25 tested and
should not be believed without an out-of-sample check.

That asymmetry is not surprising in hindsight: 2021–2026 was mostly a rising
market, so "RSI is deeply oversold" mostly selects falling knives, while "RSI
is still strong" genuinely marks bars where a bearish reversal has something
to reverse.

## Divergence: it works, but only with the outside-bar filter removed

Divergence here means price taking out the lowest low of the prior 20 bars
while RSI(21) holds above its reading at that prior low (`rsi_divergence_masks`
in `strategies.py`; bearish is the mirror).

Stacked on the outside bar it does not help — the combination is negative at
5 and 10 bars (−0.32%, −0.94%; t by ticker −2.26, −3.42), i.e. worse than
divergence alone and no better than the outside bar alone.

**On its own it is the only bullish signal in this whole study that works.**
13,795 events, positive excess at every horizon tested, and it survives both
clustering schemes (+0.24% at 5 bars, t = 3.76 by ticker and 2.73 by month;
positive on 61% of tickers and 66% of months).

Traded mechanically, the exit is what decides whether any of that is
collectable. Entry on bullish divergence, single position per ticker, exit
varied:

| exit rule | trades | win rate | avg return | excess vs random timing | t by ticker |
|---|---|---|---|---|---|
| **fixed 5-bar hold** | 11,226 | 53.9% | +0.54% | **+0.24%** | **+3.42** |
| fixed 10-bar hold | 9,467 | 53.6% | +0.79% | +0.18% | +1.38 |
| fixed 20-bar hold | 7,525 | 54.3% | +1.57% | +0.26% | +1.07 |
| fixed 60-bar hold | 4,466 | 57.3% | +4.43% | +0.49% | +1.33 |
| mirror bearish divergence (46-bar avg) | 4,658 | 69.8% | +3.30% | +0.32% | −0.30 |
| mirror, 40-bar lookback (72-bar avg) | 2,839 | 74.7% | +5.76% | +1.18% | +0.04 |

The 69.8% win rate and +3.30%/trade of the mirror-exit version is the trap
this whole document keeps running into: it is drift over a long hold. The
5-bar hold is the version that actually beats random timing, and it is the
smallest number in the table.

**Costs are not modelled anywhere in this harness, and at 5-bar holds they
decide it.** A 0.24% gross edge per trade survives a 5bp round trip and does
not survive 20bp. Adding commission/slippage modelling is already on the
README's next-steps list; until it is there, this is a signal that looks real,
not a strategy that is known to pay.

## What changed in the code

- `rsi_divergence_masks()` — the divergence primitive, with the no-lookahead
  property covered by `tests/test_rsi_divergence.py`.
- `rsi_divergence` (mirror exit) and `rsi_divergence_5d` (fixed 5-bar hold),
  both registered so the nightly run measures them.
- `outside_bar_divergence` — the "more confirmation" stack, registered
  despite testing badly, so the DB keeps accumulating evidence rather than
  the question resting on one backtest.
- `outside_bar_rsi(midline=...)` takes the tighter band directly; the bear
  side that the > 55 result belongs to still needs a short-capable engine,
  which remains the open decision from the first pass.


# What the colour tiers actually flag

Clarification from the user, and it inverts the framing of everything above:
the RSI tiers in the indicator are **warnings, not confirmation**. A bearish
engulfer printing while RSI is still above the midline is coloured to say
*this is probably a false reading*; a bullish one below the midline likewise.

That is a question about **separation**, not about each tier's own edge: does
the flagged group behave differently from the unflagged group of the same
signal? Tested as a contrast (flagged minus unflagged), clustered by calendar
month, with the sign flipped on bearish signals so that positive always means
"the signal's own direction was right".

If the flag is doing its job, the difference should be **negative** — the
flagged group should be the worse one.

| flag | 3 bars | 5 bars | 10 bars | 20 bars | 60 bars |
|---|---|---|---|---|---|
| bear, RSI > 50 | +0.10 (t 2.1) | +0.14 (t 2.2) | +0.60 (t 3.8) | +0.95 (t 3.3) | +2.37 (t 3.7) |
| **bear, RSI > 55** | **+0.11 (t 2.6)** | +0.15 (t 2.2) | +0.35 (t 2.7) | +0.52 (t 3.2) | **+2.11 (t 4.4)** |
| bear, RSI > 60 | −0.07 (t 1.9) | −0.17 (t 1.2) | −0.02 (t 1.5) | +0.09 (t 1.6) | +0.91 (t 1.7) |
| bear, RSI > 65 | −0.14 (t 0.7) | −0.48 (t −0.3) | −0.55 (t −0.9) | −0.60 (t −1.0) | +1.09 (t −0.1) |
| bull, RSI < 45 | +0.19 (t 3.1) | −0.10 (t 2.0) | −0.52 (t 1.3) | +0.40 (t 0.6) | −0.13 (t −0.2) |
| bull, RSI < 50 | +0.12 (t 2.7) | −0.04 (t 2.4) | −0.18 (t 2.5) | +0.59 (t 1.6) | +1.40 (t 1.6) |

**The bearish flag is backwards.** Bearish engulfers with RSI above 55 are not
the false ones — they are significantly the *better* ones, at every horizon
from three bars out, and the gap widens with time (+2.11% by 60 bars, t = 4.4
clustered by month). Skipping them means skipping the subset that works.

**Except at RSI > 65, where the instinct has a basis.** That group's path is
the interesting one:

| horizon | 1 | 3 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| bear signals, RSI > 65 | +0.09 | −0.23 | −0.51 | −0.71 | −0.22 | +0.89 | +1.88 |
| bear signals, RSI ≤ 65 | +0.03 | −0.09 | −0.04 | −0.16 | +0.38 | +0.69 | +0.79 |

Those go the *wrong way for about a month* — price keeps rising after the
signal, worst around 10 bars — and then resolve downward hard. If the
experience behind the colour is "I take this one and it immediately runs
against me", that is real and visible in the data. But it is 511 signals and
none of those short-horizon differences are significant (t between −0.3 and
−1.0), so it is a hypothesis worth watching, not a finding. It also means the
tier is mistimed rather than wrong: the signal is early, not false.

**The bullish flag does not separate anything.** RSI < 45 signals do slightly
*better* than the rest at 2–3 bars (t ≈ 3), slightly worse at 10, and nothing
consistent after. More to the point, both groups are net negative — flagged
−0.38% at 5 bars, unflagged −0.28% — so there is no good tier for the flag to
protect. The bullish engulfer is weak everywhere on this data, and RSI is not
the variable that splits it.

## Practical read

- Keep the bear > 55 tier, invert what it means: attention, not suspicion.
- Consider a separate tier at RSI > 65 for "right idea, early" — that is where
  the false-reading experience actually lives, and it is a timing warning
  rather than an invalidation.
- The bullish tier is not earning its place as a filter either way.


# Confirming a reversal: follow-through candles, and divergence

Three questions: can a bullish engulfer in a downtrend be trusted if one or
two bullish candles follow, does the same hold for a bearish engulfer in an
uptrend, and does RSI or MACD divergence work as the confirmation instead.

Method: the entry moves to the close of the **confirming** bar, and the
forward return is measured from there — the up bars you waited for are spent,
not counted as profit. Two controls decide what any improvement means: "wait
one bar unconditionally" separates the delay from the condition, and the same
follow-through with **no** engulfer separates the pattern from plain momentum.
Signs are flipped on bearish cohorts so positive always means the signal's own
direction was right.

## Bullish engulfer in a downtrend: confirmation makes it worse

Excess return from the entry bar, confirmed downtrend only (regime-tagged,
`confirm_bars=3`):

| entry rule | signals | 5 bars | 10 bars | t by ticker (10b) |
|---|---|---|---|---|
| engulfer, enter same bar | 2,137 | −0.29% | −0.62% | −2.21 |
| engulfer, wait 1 bar unconditionally | 2,137 | −0.21% | −0.50% | −1.73 |
| engulfer + 1 bullish candle | 1,071 | −0.26% | −0.88% | −2.82 |
| **engulfer + 2 bullish candles** | 627 | **−1.17%** | **−2.07%** | **−5.14** |
| engulfer + close above the engulfer's high | 910 | −0.04% | −0.59% | −1.85 |
| control: down bar + 2 up candles, no engulfer | 18,842 | −0.22% | −0.36% | −5.01 |

More confirmation, worse results, monotonically. Waiting a bar
*unconditionally* costs nothing — so it is not the delay, it is the condition.
Requiring up bars means only entering after a bounce has already happened, and
in a downtrend you are paying for that bounce: the higher entry price is the
entire cost, and it is bigger than the information the follow-through carries.
The control says roughly half the damage is generic bounce-chasing and the
engulfer version is worse than the control, not better.

Month-clustered t-stats here run −0.8 to −2.5, so read this as "no evidence
confirmation helps, decent evidence it hurts" rather than a law.

## Bearish engulfer in an uptrend: confirmation does help

The mirror question gives the opposite answer, which is consistent with
everything else in this document — bearish setups in this sample pay slowly
and need confirmation; bullish ones do not.

| entry rule | signals | 20 bars | 60 bars | t by ticker (60b) | t by month (60b) |
|---|---|---|---|---|---|
| engulfer, enter same bar | 3,245 | +0.70% | +2.13% | 1.82 | 1.51 |
| **engulfer + 1 bearish candle** | 1,533 | +0.95% | **+2.94%** | **3.23** | **2.17** |
| engulfer + 2 bearish candles | 707 | +1.23% | +2.12% | 1.85 | 0.94 |
| engulfer + close below the engulfer's low | 1,229 | +0.95% | +2.30% | 4.15 | 1.59 |
| control: up bar + 2 down candles, no engulfer | 22,671 | +0.43% | +1.10% | 5.45 | 0.96 |

One confirming bearish candle is the sweet spot: +2.94% over 60 bars, the only
confirmation rule in either direction that is significant under month
clustering. Two candles is worse than one — same over-waiting cost as the
bullish side, just not enough to swamp the signal. Note the horizon: nothing
happens in the first 10 bars either way. This is a slow signal.

## Divergence as the confirmation

| cohort | signals | 5 bars | 10 bars | 20 bars | 60 bars | t by month (10b) |
|---|---|---|---|---|---|---|
| bull engulfer alone | 7,139 | −0.30% | −0.35% | −0.24% | −0.23% | −1.55 |
| bull engulfer + RSI divergence | 1,148 | −0.31% | −0.97% | +0.01% | +0.42% | −0.11 |
| bull engulfer + MACD divergence | 416 | +0.02% | −0.25% | +0.73% | −0.80% | −1.11 |
| bull engulfer + RSI div + 1 up candle | 534 | −0.42% | −1.15% | −0.31% | −1.15% | −0.55 |
| **RSI divergence alone** | 13,795 | +0.24% | +0.25% | +0.65% | +1.72% | **2.37** |
| **MACD divergence alone** | 6,895 | +0.40% | **+0.93%** | **+1.38%** | +2.47% | **3.26** |
| bear engulfer alone | 8,416 | −0.06% | −0.19% | +0.34% | +0.86% | −1.30 |
| bear engulfer + RSI divergence | 1,799 | −0.06% | −0.10% | +0.91% | +2.26% | 0.61 |
| bear engulfer + MACD divergence | 743 | −0.20% | +0.10% | +1.44% | +2.69% | −1.03 |
| **bear engulfer + RSI div + 1 down candle** | 867 | +0.07% | −0.26% | **+1.41%** | **+2.75%** | −0.00 |

On the **long** side divergence does not rescue the engulfer — but divergence
*without* it remains the best signal in this whole document, and **MACD
divergence is the stronger of the two oscillators**: +0.93% at 10 bars and
+1.38% at 20 against RSI's +0.25%/+0.65%, on half as many signals. Traded on a
fixed hold, one position per ticker, against random entry timing:

| rule | trades | win rate | avg return | excess | t by ticker | t by month |
|---|---|---|---|---|---|---|
| **MACD divergence, 10-bar hold** | 5,748 | 55.3% | +1.32% | **+0.72%** | **4.48** | **2.72** |
| MACD divergence, 20-bar hold | 4,802 | 56.1% | +2.27% | +1.01% | 5.45 | 1.99 |
| RSI divergence, 5-bar hold | 11,226 | 53.9% | +0.54% | +0.24% | 3.42 | 2.60 |
| both divergences, 10-bar hold | 3,468 | 55.4% | +1.39% | +0.82% | 4.23 | 2.41 |

Requiring both divergences is not better than MACD alone — it halves the trade
count for the same edge.

On the **short** side the stack works the way the user's instinct suggested:
bear engulfer + RSI divergence + one confirming bearish candle is the strongest
bearish configuration measured (+2.75% over 60 bars, t = 4.29 by ticker), and
it is still on the side this long-only engine cannot trade.

## Standing caveats

By now dozens of cohorts have been tested on one five-year window of one
universe. Month-clustered t-stats of 2–3 in that setting are suggestive, not
established, and the multiple-comparison exposure is real. Everything here is
also gross of costs, which at 5–10 bar holds is the difference between an edge
and none. The honest next steps are an out-of-sample period (this sample is
almost entirely a rising market, which is exactly the regime that flatters
bearish-signal patience and punishes bullish reversal-chasing) and commission
and slippage modelling.

## Code

- `divergence_masks(df, oscillator, lookback)` — the primitive, now
  oscillator-agnostic; `rsi_divergence_masks` and `macd_divergence_masks` wrap it.
- `macd_divergence` (mirror exit) and `macd_divergence_10d` (fixed ten-bar
  hold, its best-measured exit), both registered.
- `tests/test_divergence.py` covers both oscillators, the no-lookahead
  property, and the fixed-hold exits.
