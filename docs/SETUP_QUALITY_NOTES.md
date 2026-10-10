# Setup quality: what the grades actually do, and what to revise

Notes as of 2026-10-10, written off the first grade history deep enough to
judge (`python -m smc_regime.grade_report`: 9,940 rows, 24 capture dates,
2026-09-01 to 2026-10-08). Nothing here is implemented. The point of
writing it down is that the next revision should start from the evidence
below rather than from an opinion about which indicator matters.

## The one result that should shape everything else

**The grade ranks at two weeks and at one month. It does not rank at one
week.**

Measured two ways, because the first way misled me. Bucket means with the
report's monotonic-ordering flag:

| horizon | A | B | C | D | A−D | ordering flag | A bucket n |
|---|---|---|---|---|---|---|---|
| 1w | −0.10 | −0.26 | −0.49 | −0.40 | +0.30 | broken (D > C) | 607 |
| 2w | **+0.38** | +0.03 | −1.23 | −1.51 | **+1.89** | holds | 420 |
| 1m | −1.59 | −0.82 | −2.24 | −3.01 | +1.42 | broken (B > A) | 89 |

And by rank correlation between the raw score and the forward return,
computed per capture date (`smc_regime/grade_metrics.py`):

| horizon | pooled IC | per-date IC | dates positive |
|---|---|---|---|
| 1w | +0.039 | +0.044 ± 0.098 | 11/19 (58%) |
| 2w | **+0.113** | +0.118 ± 0.086 | **13/14 (93%)** |
| 1m | **+0.103** | +0.113 ± 0.045 | **5/5 (100%)** |

**The two disagree at one month, and the ordering flag is the one that is
wrong.** It trips on a single A-vs-B inversion across an 89-row A bucket,
while the rank correlation — which uses all 2,055 matured rows rather than
four bucket means — is +0.103 and positive on every capture date measured.
The signal at 1m is about as strong as at 2w. An earlier version of this
note said the grade "works at two weeks and only there"; that was reading a
brittle statistic.

1w is the genuinely weak horizon on both measures: the spread is +0.30%,
inside the noise, and the IC is positive on barely half the dates.

Two things follow.

1. **The page should state the horizon.** Nothing on the dashboard says
   what a grade predicts. It predicts a hold of roughly two weeks to a
   month. A reader taking an A-graded setup for three days is using a
   number measured not to work over three days.
2. **Do not track the ordering flag.** One noisy bucket flips it, so it
   would flicker between runs while the underlying signal sat still. Track
   the per-date IC, which degrades gracefully and has a dispersion you can
   reason about. This is now `grade_metrics`.

## Grades rank probability better than magnitude

Win rate is monotonic at both 1w (46.0 / 42.3 / 40.0 / 38.5) and 2w, while
*mean return* is monotonic only at 2w. The grade is better at saying "this
one is more likely to work" than "this one will work bigger".

That is a useful thing to know and an argument against using the score as a
position-sizing input, which is the obvious next thing someone would reach
for. It is an argument FOR using it as a selection rank — which is exactly
the gap the portfolio work hit (see below).

## Revisions worth testing, roughly in order of expected value

### 1. Expose a price-only sub-score

Six of the eight components are computed from price alone and could be
reconstructed for any historical bar:

    trend structure 20 + RSI 15 + MACD 15 + volume 15 + streak 10
      + daily/weekly alignment 10  =  85 points

The other two — sector/industry alignment (5) and valuation (10) — need
external data that cannot be honestly dated backwards. Yahoo serves
*current* forward P/E; there is no historical series, which is why
`grade_backfill` scores valuation at neutral half credit and why 2,055 of
the 9,940 rows in the report are not like-for-like with the rest.

This is not a cosmetic split. The portfolio work needed a per-signal
quality score to rank candidates for scarce slots, and **the grades could
not be used for it** precisely because the valuation and sector terms would
have leaked present-day information into a 2019 backtest. A price-only
sub-score would be backtestable over the full history and droppable
straight into `portfolio.simulate(selector="ranked", scores=...)`. It
connects the two halves of this project, which currently do not speak.

### 2. Make the cuts percentile-based

The cuts (A ≥ 76, B ≥ 68, C ≥ 57) were calibrated once, against this
universe's spread on the first run with real technicals, targeting roughly
A 10% / B 25% / C 40% / D 25%. They have drifted:

| | now | target | drift |
|---|---|---|---|
| A | 15.9% | 10% | +5.9pp |
| B | 36.4% | 25% | +11.4pp |
| C | 34.5% | 40% | −5.5pp |
| D | 13.3% | 25% | −11.7pp |

Half the board is now A or B against a 35% target. Either the universe
genuinely improved or the cuts no longer describe it, and a fixed threshold
cannot tell you which. Percentile cuts self-calibrate and make the letter
mean the same thing across time — at the cost of never being able to say
"the whole board is weak this week", which a fixed cut CAN say. Worth
having both: percentile for the letter, absolute score kept visible.

### 3. Test whether valuation earns its 10 points

Forward P/E predicting a *two-week* return is implausible on its face, and
it is already neutralised for 21% of the history. The test is cheap and
uses data in hand: recompute the grade without the valuation term, re-run
`grade_report`, compare the 2w spread and ordering. If the spread is
unchanged or better, that is 10 points — a tenth of the whole scale —
available to reallocate to something that does predict at this horizon.

Same test applies to sector/industry alignment (5 points), though less
urgently since it is cheaper to compute and does not need a backfill fudge.

### 4. There is no risk term anywhere in the grade

An A on a 5%-ATR name and an A on a 1%-ATR name are the same letter and a
completely different proposition. Across this project's measurement work
the one finding that has survived repeatedly is that **risk control works
and return prediction does not** — stops cut the sub-−50% tail from 20.7%
to 0.2% for 0.054pp of excess, diversification cut drawdown −38.8% → −33.7%,
and a portfolio quality floor beat SPY's drawdown in 20 of 20 cells while
beating its return in none.

A grade that says nothing about volatility is leaving the one measurable
edge on the table. Adding realised vol or ATR% would not improve the
ranking of returns; it would let the grade speak to sizing, which is where
the evidence says the value is.

### 5. Only RSI is regime-conditional

`_rsi_points` reads RSI *against* the regime — a pullback into the 30s
inside a trend is an entry, the same reading at the top of a choppy range
is not. That is the most defensible piece of logic in the scorer. But MACD,
volume and trend structure are scored identically in every regime. Whether
regime-conditioning those improves the 2w spread is a direct test.

### 6. The letter is noisy at the cuts, and `~A` is a patch over that

Scores bunch within a point of the boundaries — the A band alone averages
under 0.4 points between adjacent names — which is why borderline rows
carry a dashed `~A` / `~B` tag. That tag is a workaround for a scale
problem: a one-letter output cannot carry "87.5 and 87.4 are the same
setup". Reporting a percentile rank alongside the letter would fix the
underlying issue rather than annotate it.

## Guard rails for whoever does this

**Twenty-four capture dates is not enough to fit weights to.** The
portfolio work in this repo spent a full session learning this: a
pre-registered rule that maximised in-sample return picked the worst
configuration in 5 of 5 folds, a quality floor that looked like a +31%
edge in one window turned out to be an exposure dial that lost 22% in the
next, and the one effect that survived was +1.57pp. Eight weights fitted to
9,940 rows drawn from 24 dates and one market regime would overfit
comprehensively and look excellent doing it.

Test revisions ONE AT A TIME against the 2w and 1m per-date IC, keep the
horizons and the metric fixed while doing it, and prefer a change that is
justified mechanically (valuation cannot predict two weeks) over one that
is justified by a better backtest number.

**Segment the backfilled rows before drawing conclusions.** The report
flags the 2,055/7,885 split but still pools them. Any comparison that turns
on valuation especially needs the live-only cut, since that is the term the
backfill neutralises.
