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

## Measured: which component earns its weight

Two measurements, 2026-10-10. Neither is conclusive and they partly
disagree, which is the main thing to carry forward.

### Standalone IC of each deployed component (grade history, 14 dates @ 2w)

| component | weight | 2w IC | hit | 1m IC | hit |
|---|---|---|---|---|---|
| trend_structure | 20 | **+0.143** | 100% | **+0.127** | 100% |
| macd | 15 | +0.113 | 86% | +0.035 | 60% |
| alignment | 10 | +0.052 | 79% | +0.033 | 60% |
| volume | 15 | +0.024 | 43% | +0.018 | 60% |
| streak | 10 | +0.012 | 50% | **−0.073** | 0% |
| sector_industry | 5 | −0.056 | 36% | +0.079 | 100% |
| rsi | 15 | **−0.066** | 29% | +0.001 | 40% |
| valuation | 10 | **−0.131** | 11% | — | — |
| *whole grade* | 100 | +0.118 | 93% | +0.113 | 100% |

**14 overlapping dates is 1–2 independent observations.** A "100% hit rate"
on that is close to meaningless. Only trend_structure's adjusted t (+2.6)
approaches significance and it would not survive a multiple-comparison
correction across eight components.

### Incremental: does the grade rank better WITHOUT a component?

| variant | pts | 2w IC | Δ | 1m IC |
|---|---|---|---|---|
| full grade | 100 | +0.1180 | — | +0.1123 |
| drop volume | 85 | +0.1200 | +0.002 | +0.1138 |
| drop streak | 90 | +0.1219 | +0.004 | +0.1413 |
| drop rsi | 85 | +0.1259 | +0.008 | +0.0973 |
| drop sector_industry | 95 | +0.1270 | +0.009 | +0.0970 |
| drop valuation | 90 | +0.1427 | +0.025 | +0.1123 |
| price-only (drop sec_ind+val) | 85 | +0.1506 | +0.033 | +0.0970 |
| trend+macd+alignment only | 45 | +0.1700 | +0.052 | +0.1183 |

**Every removal improves the 2w IC, and that is a reason for suspicion
rather than a mandate to delete half the scale.** Summing non-predictive
components adds variance to the composite, so dropping any noise term
raises IC mechanically whether or not the term is harmful. This table
cannot separate dead weight from active harm. The horizons also disagree —
dropping rsi helps at 2w and hurts at 1m; dropping streak is the best 1m
variant and does nothing at 2w — which is what fitting noise looks like.

**Do not adopt the 45-point variant.** Selecting the best of eleven
variants on 1–2 effective observations is the exact failure the portfolio
work in this repo spent a session documenting.

### The one change with evidence from two directions

**Valuation.** Standalone IC −0.131, positive on 11% of dates; the largest
single incremental gain (+0.025); and an a priori reason that needs no
statistics — a forward P/E cannot plausibly rank a two-week return, and the
backfill already neutralises it for 21% of history. Three independent
arguments, one direction. This is the safe subtraction.

### Volume: the signed/unsigned finding

Measured on the full 767k-row panel (1,885 dates), not the 24-date history,
so this one IS well powered:

| rung | 2w | 1m |
|---|---|---|
| 5v50 *(deployed)* | +0.0086 | +0.0082 |
| 30v50 | −0.0014 | −0.0033 |
| 10v30 | +0.0087 | +0.0076 |
| 3v10 | +0.0004 | +0.0025 |
| cascade (all three building) | +0.0037 | +0.0037 |
| **any version signed by price direction** | **−0.004 … +0.001** | **negative** |

Three results. A nested ladder (30v50 / 10v30 / 3v10) adds nothing: only
the middle rung works and it merely ties the 5v50 already deployed, while
both ends are indistinguishable from zero. Combining rungs dilutes — the
cascade (+0.0037) scores below its own best rung (+0.0087), the same
averaging-away-the-signal result the multi-timeframe trend rungs showed.
And **signing volume by price direction turns a positive signal negative**,
on every rung and both timeframes.

That last one is actionable, because the deployed component scores exactly
that product — "advance on heavy volume → conviction". Scoring volume
UNSIGNED is a cheap, well-powered change. Weekly volume carries nothing at
all (best rung +0.005, several negative), unlike weekly *trend*.

### Trend structure: the two measurements disagree 14-fold

Deployed component over 14 dates: +0.143. A price-only reconstruction of
the same idea over 1,944 dates and 7 years: +0.010. The gap is sample
(six weeks vs seven years) and fidelity (real scorer vs approximation), and
nothing here resolves it. Treat both as provisional.

On the same 7-year panel, of eight richer trend features only the WEEKLY
EMA stack beat the current 50/200 read (+0.0135/+0.0158 vs +0.0102/+0.0099).
Swing structure (HH/HL vs LL/LH) measured +0.0018 and +0.0000 — the weakest
of the eight, and the strict consecutive-run variant was barely better.
Rung aggregation again scored below its best single rung, though strict
unanimity beat a graded count.

### A detail worth fixing while in here

Components are stored rounded to 1dp while `total_points` is rounded from
the unrounded sum, so summing the stored components differs from the stored
total by up to 0.2 points on 41% of rows. Harmless for the IC, but adjacent
A-band scores differ by under 0.4 points, so that jitter does reorder names
near the cuts — the same tight-bunching the `~A` tags exist for.

## SETTLED: the full-panel measurement (2026-10-10)

`smc_regime/price_score.py` reconstructs the six price-only components for
every bar, so these run on **1,944 dates and 767,147 rows** instead of 24
dates and 9,940. This is what the 14-date tables above could not do, and it
contradicts most of them.

### Per-component IC, full panel

| component | 2w IC | 1m IC | hit | t_adj 2w | *(14-date claim)* |
|---|---|---|---|---|---|
| trend_structure | **+0.0140** | **+0.0149** | 54/57% | +1.0 | *+0.143* |
| alignment | +0.0098 | +0.0127 | 54/57% | **+1.3** | *+0.052* |
| rsi | +0.0038 | +0.0019 | 49/50% | +0.4 | *−0.066* |
| streak | +0.0027 | +0.0011 | 53/51% | +0.5 | *+0.012* |
| macd | −0.0040 | −0.0041 | 49/49% | −0.4 | *+0.113* |
| volume | −0.0040 | −0.0056 | 48/49% | −0.6 | *+0.024* |
| **price_score** (85 pts) | +0.0103 | +0.0087 | 55/54% | +1.1 | — |

**RSI IS NOT INVERTED.** The −0.066 with a 29% hit rate was sampling error;
on 1,933 dates it is +0.0038, indistinguishable from zero. The planned
"is RSI backwards" investigation is closed, and the −0.38 correlation with
trend_structure is not evidence of cancellation — it is two roughly
uncorrelated-with-returns signals that happen to co-move.

**MACD's apparent strength was also noise**: +0.113 on 14 dates, −0.0040 on
1,933. So was trend_structure's +0.143; its real value is +0.0140, an
order of magnitude smaller and close to the +0.0102 the independent
price-only proxy measured earlier. That 14-fold disagreement is now
resolved in favour of the large sample.

### Sub-periods, 2w — the stability test

| component | 2019-20 | 2021-22 | 2023-24 | 2025-26 | sign flips |
|---|---|---|---|---|---|
| **trend_structure** | +0.0189 | +0.0139 | +0.0065 | +0.0197 | **0** |
| alignment | +0.0166 | +0.0156 | −0.0007 | +0.0087 | 2 |
| rsi | +0.0224 | −0.0025 | −0.0065 | +0.0016 | 2 |
| streak | +0.0050 | +0.0082 | −0.0082 | +0.0062 | 2 |
| macd | −0.0180 | +0.0022 | +0.0015 | −0.0015 | 2 |
| volume | +0.0013 | −0.0104 | +0.0006 | −0.0074 | 3 |
| price_score | +0.0123 | +0.0184 | −0.0019 | +0.0127 | 2 |

**trend_structure is the only component positive in all four eras.**
Everything else changes sign at least twice. Four-for-four under a null of
random signs is p=0.125, so this is a pattern rather than a proof, but it
is the only consistency in the table and it agrees with trend_structure
also being the largest IC and the most costly to remove.

### Incremental, full panel — and it reverses the 14-date result

| variant | pts | 2w IC | delta | 1m delta |
|---|---|---|---|---|
| price-only (all six) | 85 | +0.0103 | — | — |
| drop trend_structure | 65 | +0.0031 | **−0.0072** | −0.0070 |
| drop alignment | 75 | +0.0070 | **−0.0033** | −0.0041 |
| drop rsi | 70 | +0.0074 | −0.0029 | −0.0027 |
| drop streak | 75 | +0.0111 | +0.0009 | +0.0012 |
| drop macd | 70 | +0.0128 | +0.0026 | +0.0040 |
| drop volume | 70 | +0.0132 | +0.0029 | +0.0036 |

On 14 dates EVERY removal improved the score and the table was
uninterpretable. With 1,900 dates three removals clearly hurt and three
mildly help, which is a real ranking rather than the variance artefact.
Dropping trend_structure costs more than twice what any removal gains.

Note the composite still ranks BELOW its best component (+0.0103 against
trend_structure's +0.0140), now at full power. The earlier diagnosis holds:
the components are near-independent (6.9 effective of 8), so the two with
negative IC subtract rather than dilute. macd and volume are those two, and
they are exactly the two whose removal helps.

### The sobering part

**More data did not produce significance.** With 767,147 rows the largest
overlap-adjusted t is +1.3. The effects are real-looking but tiny: an IC of
0.01 is economically trivial, and the honest reading is that the price-only
grade ranks forward returns barely better than chance. The reconstruction
succeeded as an instrument -- it settled which 14-date findings were noise,
which was its purpose -- without turning up a signal worth acting on.

Caveat carried from validation: `alignment` reconstructs at only 0.80
correlation with the deployed component (it is a three-valued step
function, so any regime disagreement swings it 5 points). Its numbers here
describe the reconstruction. The other five correlate 1.00.

### What this changes in the recommendations above

- **Drop the "is RSI inverted" investigation.** Answered: no.
- **The valuation cut still stands**, on the a priori argument and the
  backfill neutralisation, neither of which this panel touches (valuation
  is not price-only and cannot be reconstructed).
- **Unsigning volume still stands** -- measured on its own well-powered
  panel, and volume is one of the two negative-IC components here.
- **Do not pursue component reweighting.** At IC 0.01 with t_adj ~1, there
  is no weighting of these six that produces a useful score.

## A DIFFERENT TARGET: does the grade predict that the regime CONTINUES?

Everything above scores the grade against forward returns. A regime tool is
arguably for a different question -- at bar t, is the confirmed (regime,
direction) still the same N bars later? -- and the grade is built largely
from persistence-flavoured inputs, so it deserved its own measurement
rather than an inference from the return work. `analysis/persist.py`,
763,027 observations.

### There is no continuation probability to predict

| horizon | same regime+direction | same direction only |
|---|---|---|
| 2 weeks | **49.3%** | 50.8% |
| 1 month | **37.2%** | 38.1% |

Direction alone survives two weeks 50.8% of the time. That is a coin flip,
and it is the efficient-market answer rather than a defect in the
classifier: "will this trend continue" has no edge in it at this horizon,
for anyone, before any scoring is applied.

### Two findings about the classifier, both usable

| regime | persists 2w | n |
|---|---|---|
| choppy | 55.7% | 241,573 |
| trending | 47.0% | 511,821 |
| **parabolic** | **6.7%** | 9,633 |

**Parabolic persists 6.7%.** By the time a move carries that label it is
essentially over. The label should be read as "this is ENDING", never as
"this is happening" -- close to the opposite of how a parabolic tag is
normally used, and worth saying on the page.

**Trending persists LESS than choppy**, 47.0% against 55.7%. The regime a
reader would most want to extrapolate is the least stable one.

### The grade adds nothing on top of that

AUC, where 0.50 is no separation. Pooled is shown only to be dismissed:
trending and choppy persist at different rates and the grade scores
differently across them, so a pooled number can look predictive while
merely re-reading the regime label the caller already has.

| feature | pooled 2w | choppy | trending |
|---|---|---|---|
| price_score | 0.4903 | 0.4733 | 0.5070 |
| rsi | 0.4123 | 0.5254 | 0.4242 |
| macd | 0.5204 | 0.4675 | 0.5468 |
| alignment | 0.5114 | 0.5000 | 0.5190 |
| volume | 0.4961 | 0.4972 | 0.4950 |
| streak | 0.4995 | 0.4877 | 0.4922 |

`price_score` is 0.4903 pooled -- below chance. Three features (rsi, macd,
price_score) flip sign between regimes in near-mirror image, which is the
shape of noise rather than regime-conditional information.

Persistence by score quartile, within regime, 2w:

    choppy    base 55.7%   Q1 57.4  Q2 58.0  Q3 56.8  Q4 50.7   Q4-Q1 -6.6pp
    trending  base 47.0%   Q1 49.5  Q2 44.3  Q3 42.6  Q4 51.8   Q4-Q1 +2.3pp

Choppy is INVERTED -- the best-scored quartile persists 6.6pp less -- and
trending is U-shaped. Neither is monotonic, which is what a real
relationship would look like and what noise does not.

### The cleanest falsification of the day: `streak`

**`streak` scores AUC 0.4877 in choppy and 0.4922 in trending. Both below
0.5.**

The component awards 10 of the grade's 100 points for a long regime
streak, on the stated premise that a regime which has held is more settled
and therefore more likely to continue -- `regime_streak_bars`' own
docstring says "a longer streak there means the regime has held (not just
been confirmed once), a stronger signal". Measured across 750,000 bars,
how long a regime has already run carries NO information about whether it
survives the next ten bars, and leans very slightly the wrong way.

Unlike the component findings from the 24-date history, this one is well
powered. It is a documented rationale contradicted by data, and it is the
most concrete thing on this page: 10 points are being awarded for a
property that has been measured not to exist.

### What this adds to the recommendations

- **Do not build a "probability the trend continues" feature.** The
  underlying quantity is 49-51%, before any model.
- **Say what parabolic means on the page.** 6.7% persistence makes it an
  exit signal, not a state.
- **`streak` is now a third candidate for subtraction**, alongside
  valuation and the volume signing -- and it is the best evidenced of the
  three for the specific claim it makes.

## Correction: the break-even cost figure was wrong by 2.2x

Recorded because it was quoted repeatedly before being checked, and
because the error has a shape worth recognising.

The claim was "break-even at 40.8bp per side". A basis point is 0.01%, and
"per side" means charged on EACH fill, so a round trip pays twice and
break-even per side is half the per-trade excess. Recomputed over 37,037
rsi2_prior_high trades against a same-ticker same-duration baseline:

    gross per trade    +0.7590%
    baseline           +0.3848%
    excess             +0.3742%  = 37.4bp
    break-even          37.4bp round trip  =  18.7bp PER SIDE

So 40.8bp was a round-trip number wearing a per-side label. The tell was
available without recomputing anything: 40.8bp is within rounding of the
+0.411% excess that was quoted beside it, and an excess cannot equal its
own per-side break-even.

**Turnover is what actually decides it, and it was never mentioned.** Mean
hold is 4.1 bars -- roughly 60 round trips per slot per year:

    cost/side   net per trade   annual drag per slot
          0bp        +0.3742%                  0.0%
          5bp        +0.2742%                  6.1%
         10bp        +0.1742%                 12.2%
         20bp        -0.0258%                 24.4%

A 37bp edge does not survive being paid sixty times. This reconciles with
the walk-forward -- beats SPY at 0bp, mixed at 5bp, loses every fold at
10bp -- which had sat in the SAME docstring as the 40.8bp claim without the
inconsistency being noticed.

`rsi2` is slower (7.9-bar hold, ~32 round trips/year) and its excess is
smaller: 25.3bp round trip, 12.7bp per side.

And the benchmark matters as much as the arithmetic. 18.7bp is break-even
against holding the same name for the same bars. Against simply buying
SPY it is lower, because that baseline supplies market beta with no
turnover at all.

Derivation lives in `analysis/breakeven.py` so the number is checkable
rather than quoted.

## Exit policies: trailing stops, trimming, averaging down

61,795 trades across five trend strategies (ema_cross, macd,
rsi_dip_recovery, supertrend, donchian), mean hold 35 bars, each replayed
bar by bar. `analysis/exitpol.py`. Stops fill at min(open, stop) per the
repo convention; when a target and a stop both hit in one bar the stop is
assumed first, since daily bars cannot resolve intraday order.

| policy | mean % | tail kept | cut early | beats base |
|---|---|---|---|---|
| baseline | +3.385 | 100% | — | — |
| fixed5 | +1.697 | 53.2% | 16.2% | 1/4 |
| **trail5** | **+0.665** | **13.4%** | **31.6%** | 1/4 |
| fixed20 | +2.942 | 92.1% | 2.3% | 0/4 |
| trail20 | +2.329 | 73.0% | 5.4% | 1/4 |
| **be30** | **+3.267** | **96.2%** | **0.4%** | 1/4 |
| be10 | +2.614 | 76.5% | 4.8% | 1/4 |
| trim30 | +2.784 | 80.2% | 5.2% | 1/4 |
| **avgdn10** | **+4.249** | 103.2% | 0% | **4/4** |

**TRAILING STOPS ARE WORSE THAN FIXED AT EVERY WIDTH**, which refutes the
prior stated before the run. A 5% trail keeps 13.4% of the baseline's
right tail and cuts a third of trades early. The mechanism: a trend's
normal pullback IS a 5-10% retrace from the running high, so a trailing
stop lives inside the noise, while a fixed stop sits below entry and never
gets dragged up into it.

**BREAKEVEN STOPS ARE THE ONE CHEAP POLICY.** be30 costs 0.12pp of mean
and keeps 96.2% of the tail; be10 gives the table's largest median
improvement (-1.151% to -0.450%). They are cheap precisely because they
only arm AFTER the trade is already up.

**NOTHING EXCEPT AVERAGING DOWN BEATS BASELINE IN MORE THAN 1 OF 4 ERAS**,
and every single-era win lands in 2021-22 -- the one period where the
baseline loses money. That is insurance, not edge, and it is the same
shape as the fixed-stop result from the morning.

### Averaging down: better mean, roughly double the tail

Only 4/4 policy, and it survives the capital adjustment: +3.631% per unit
of capital against baseline's +3.385%. Then the left tail:

| | baseline | avgdn10, capital-weighted |
|---|---|---|
| worst trade | -98.4% | **-196.6%** |
| p1 | -25.48% | **-43.12%** |
| on the 10,525 it added to | mean -3.58% | **+3.00%** |
| their p05 | -26.92% | **-46.15%** |
| share still ending negative | — | **74.1%** |

It converts a -3.58% average into +3.00% on the trades it touches, but
**74% of those still lost** -- it is not rescuing most of them, the few it
rescues outweigh in the mean. The 1-in-100 outcome goes -25% to -43%.
Textbook martingale: better average, better win rate, roughly 2x tail.

Doubly flattered: the trade-level replay cannot charge the capital tied up
at portfolio level, and 2019-2026 contains only V-shaped recoveries. A
prolonged or terminal decline is exactly what this sample lacks.

## Fixed brackets on rsi2: a scale mismatch

`analysis/bracket.py`, `analysis/bracket2.py`. +20%/-10% and +10%/-5%,
both 2:1 so both need a **33.3% hit rate** to break even.

| | WITHIN mean | hit rate of decided | PURE mean | edge over RANDOM entry |
|---|---|---|---|---|
| rsi2 20/10 | +0.763% | 6.6% | +3.532% | **+0.376pp** |
| rsi2 10/5 | +0.483% | 14.3% | +1.210% | **+0.146pp** |
| rsi2_ph 20/10 | +0.532% | 4.1% | +3.436% | +0.280pp |
| rsi2_ph 10/5 | +0.337% | 10.5% | +1.152% | +0.088pp |

*(baselines +0.983% / +0.759%)*

**WITHIN mode fails at both widths and the asymmetry is structural.** RSI(2)
buys INTO a decline, so the downside level is touched long before the
upside one: at 10/5 the stop fires on 34.7% of trades and the target on
5.8%. You pay for a stop and collect a target you never reach.

**PURE mode is mostly beta.** Holding until a bracket resolves takes 58
bars at 20/10 and 19 at 10/5. A RANDOM-ENTRY control in the same names
over the same period returns +3.156% (20/10) and +1.064% (10/5), so the
signal contributes +0.09 to +0.38pp and the rest is market plus a survivor
universe. Tightening the bracket SHRINKS the edge over random.

**A design flaw worth recording**, caught by noticing the trade counts were
identical across widths: the bracket is a post-processing replay over a
fixed trade list, so PURE mode counts overlapping positions a
single-position-per-ticker engine could never hold. Only 41% (rsi2) and
38% (rsi2_prior_high) of signals are actually takeable. De-overlapping
barely moves the per-trade mean (+3.675% vs +3.532%), so the conclusion
survives, but the trade COUNT was misleading.

**The root problem is scale.** RSI(2)'s edge is +0.4% to +1.0% per trade
over 4-8 bars. A 10% target is ten times that, a 20% target twenty-five
times. The only way to reach it is to hold long past the point the signal
means anything -- at which point you are holding the market, and the
control proves it.

## Options: plausible, unprovable, and IV is the whole game

Measured from live chains 2026-10-09 (TradingView), applied to the real
rsi2 trade distribution with Black-Scholes repricing at exit.

| pricing | premium %spot | mean | median | win% | **ret/vol** | top-decile share |
|---|---|---|---|---|---|---|
| SPY IV 13.6% | 1.85% | +89.8% | +50.6% | 64.3% | **0.420** | 61.9% |
| NVDA IV 31.1% | 3.97% | +18.6% | +12.7% | 58.3% | **0.191** | **116.8%** |
| single IV 45% | 5.65% | +3.7% | +4.5% | 53.9% | **0.053** | 374.7% |
| *the stock* | — | +0.98% | +1.86% | 69.1% | **0.132** | — |

A 2:1 payoff IS mechanically reachable: an ATM call is ~28x notional
leverage, so the median trade's +1.86% spot move becomes +12.7% on the
option at realistic single-name IV. Risk-adjusted it beats the stock
(0.191 vs 0.132) because convexity truncates the left tail -- max loss is
the premium, against the stock's -10.8% p05.

**But IV decides everything**: +89.8% at index vol, +18.6% at NVDA's 31%,
+3.7% at 45%, and above ~40% IV the option is WORSE risk-adjusted than the
stock. And at realistic pricing the top decile contributes **116.8%** of
total P&L, meaning the other 90% of trades collectively lose.

**Two corrections to earlier claims in this session.** Option spreads were
asserted at ~80bp per side and "4x the stock break-even". Measured: SPY
15bp, NVDA 87bp -- wrong for the index, roughly right for a single name,
and in neither case the binding constraint. Separately, a first pass used
a delta+gamma Taylor expansion, which is only valid for small moves and
overstated a 20% move's payoff by ~58%; it was replaced with proper
Black-Scholes repricing.

**Why this cannot be settled here.** No historical options data is
available on these plans (Alpha Vantage HISTORICAL_OPTIONS is premium;
Massive's price and quote endpoints are not entitled; TradingView serves
current expirations only). Every figure above comes from ONE October 2026
snapshot applied to seven years of trades, and 2022 single-name IV ran far
above it. The model also holds IV constant entry to exit, while real IV
falls when a stock rallies -- a bias that runs one way only, against long
calls, on exactly the winning trades.

The version of the idea that does not need the forecast is **defined
risk**: a long option caps loss at the premium with no gap risk. That is
the tail control this project keeps measuring as the thing that works, and
it does not require IV to cooperate.

## Covered calls written into a drawdown

`analysis/coveredcall.py`, 734,187 ticker-bars. The idea: a position is
down, you do not want to sell, so you write a call to blunt the loss. The
tension is that in a drawdown you are waiting for a recovery and a covered
call sells exactly that recovery.

Premium is Black-Scholes priced off each name's own TRAILING 20-day
realised vol times an implied-over-realised multiplier. Using trailing
realised vol matters: vol rises in drawdowns, so the premium rises exactly
when the call is written, which is the real behaviour.

Writing +5% calls, 21-bar hold:

| drawdown when written | hold p10 | CC p10 | hold p25 | CC p25 | hold <0% | CC <0% |
|---|---|---|---|---|---|---|
| none (> -5%) | -8.99 | -7.05 | -3.68 | -2.19 | 43.4% | 35.6% |
| mild -5..-15% | -10.93 | -8.18 | -4.64 | -2.40 | 43.9% | 34.4% |
| deep -15..-30% | -14.61 | -9.81 | -5.96 | -1.92 | 42.6% | 30.2% |
| **severe < -30%** | **-21.90** | **-13.14** | **-9.82** | **-1.84** | 43.7% | **28.2%** |

**IT WORKS BETTER THE DEEPER THE DRAWDOWN**, which is the opposite of most
things tested here, and the mechanism is simple: vol rises with the
drawdown so the premium collected rises with it -- 9.07% of spot at severe
against 1.89% at none. At severe drawdown p10 improves 8.8pp and p25 by
8.0pp.

**BUT READ WHAT IS ACTUALLY HAPPENING.** The premium shifts the whole
distribution up ~9pp and caps it at +5%. The improvement in "share ending
negative" (43.7% -> 28.2%) is mostly that level shift: any outcome between
-9.07% and 0 now prints positive. That is the distribution moving, not
risk being removed. It is paid for by surrendering everything above the
strike, worth ~9.4pp in expectation at severe drawdown -- which is why the
mean barely moves. Options are priced to make this roughly fair.

### The mean effect is the volatility risk premium, and it was assumed

Mean delta vs just holding, by implied-over-realised multiplier:

| IV mult | strike | deep DD | severe DD |
|---|---|---|---|
| 1.00 | +5% | -0.740 | -1.629 |
| 1.00 | +10% | -0.405 | -1.157 |
| 1.05 | +5% | -0.468 | -1.181 |
| **1.15** | +5% | **+0.078** | **-0.301** |
| 1.25 | +10% | +0.812 | +0.957 |

**The sign flips around 1.10.** Priced at realised vol with no risk
premium, covered calls COST money on the mean. Every positive figure in
the first table is the VRP, which this work assumed at 1.15 rather than
measured. The VRP is real and well documented, but its magnitude is an
input here, and it is compensation for being short volatility -- you lose
in the crashes where the protection is most wanted.

**What survives the assumption**: the downside improvement, because that is
the premium level-shift and happens at any multiplier. The mean effect
does not survive it.

So: a risk transfer at roughly fair value, not an edge. The same shape as
stops, diversification and the cash floor -- and like those, genuinely
useful to anyone who values the smoother path, which is what the question
was actually asking.

### Not modelled

American early assignment (likely around dividends, which would cut the
position), option transaction costs on a monthly roll, and the overlapping
windows -- every bar is an observation, so the effective sample is far
smaller than 734,187. Survivorship applies as everywhere else.

## A note on 40-50 position portfolios

The portfolio work found seed dispersion falling from +-4.07pp to +-0.80pp
going 10 -> 40 slots. That is not a realistic retail book, and it cuts at
the usefulness of the finding. At 8-10 positions the outcome is dominated
by WHICH names were drawn rather than by the strategy, which is the regime
where a working system and a lucky one cannot be told apart.

Two things follow. Concentration does not lower expected return, it raises
variance -- a choice rather than an error for someone who can hold through
it and is not trying to prove a method works. And since the 40-slot book
converged on the index anyway (+0.45pp, -14.9% dd vs SPY's -14.1%), the
practical form is to take diversification from an index and spend the
effort on the levers that do not need 40 names: costs, sizing, tail
control.

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
