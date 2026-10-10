# Portfolio walk-forward

Three constructions of the same signal, run across five sequential
out-of-sample folds (2022-2026). The findings are recorded in
`smc_regime/portfolio_cli.py`'s module docstring; the CSVs here are the
raw per-cell output behind them.

| script | what it answers |
|---|---|
| `walk_forward_selected.py` | the pre-registered select-then-evaluate rule, applied per fold |
| `walk_forward_fixed.py` | no selection at all -- every slot count simply reported |
| `walk_forward_ranked.py` | ranked selection and a quality floor vs the random baseline |

Short version: the selected rule picked 10 slots by fallback in 5 of 5
folds and lagged SPY by 5.52pp on average; the fixed sweep shows the gap
AND the drawdown converging to SPY's as the book widens to 40 slots
(because 40 equal-weighted names from 415 large caps is the index with
tracking error); and of the 48 ranked/floor cells in the two clean
trending folds, zero beat SPY. Depth-ranking is worth +1.57pp, positive
in all five folds but an order of magnitude smaller than the ±20-36pp
tie-break noise it would have to overcome to justify concentrating.

## Running them

They read two pickles from a cache directory -- `universe_1d.pkl`
(`{ticker: OHLCV DataFrame}`) and `wf_trades.pkl` (written by
`walk_forward_selected.py`, reused by the other two so the trade set is
identical across all three). The cache is NOT committed: it is ~37MB of
re-fetchable price data, and committing it would do to this repo what
`backtest_logs/smc_regime.db` was excluded to avoid.

```bash
export SMC_CACHE=/path/to/cache          # defaults to analysis/cache
python analysis/walk_forward_selected.py   # also writes wf_trades.pkl
python analysis/walk_forward_fixed.py
python analysis/walk_forward_ranked.py
```

To build `universe_1d.pkl`, fetch each ticker in
`smc_regime/tracking_universe.txt` with
`smc_regime.cross_validate.fetch_yahoo_ohlcv(ticker, "2019-01-01")` and
pickle the dict. Note that Yahoo's adjusted closes differ run to run in
about the 7th significant figure, so a refetch moves every figure
slightly -- that is fetch noise, not a discrepancy. The committed CSVs
were produced on bars through 2026-09-25.

## Grade-component measurement

| script | what it answers |
|---|---|
| `trendfeat.py` | do richer trend reads (EMA stack, HH/HL swing structure, multi-timeframe agreement) rank better than the current 50/200 read |
| `compvol.py` | which grade component earns its weight, and does volume read better as a nested ladder (30v50 / 10v30 / 3v10) across 1d and 1wk |
| `incr.py` | does the grade rank BETTER without a given component |

Findings are in `docs/SETUP_QUALITY_NOTES.md`. Headlines: only the weekly
EMA stack beat the current trend read, swing structure measured ~zero, rung
aggregation scored below its best single rung in every test, and signing
volume by price direction turns a positive signal negative. `incr.py` shows
every removal improving the 2w IC, which is a warning about the metric
rather than a mandate — dropping any non-predictive term raises a composite
IC mechanically.

`trendfeat.py` and `compvol.py` read `universe_1d.pkl` from the cache
directory (see above); `incr.py` and part A of `compvol.py` read the
committed grade history and need no cache.
