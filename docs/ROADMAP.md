# Roadmap: from daily tracking to a swing-trading reference tool

Status as of 2026-08-19: **build phase, in progress.** Data source migrated
to Tiingo (paid tier). SQLite trade-level store (`backtest_logs/smc_regime.db`)
and `ticker_metadata` (exchange + GICS sector, all 132 tracked tickers
resolved) are live; `recommend_cli` now queries SQLite with the
symbol -> sector -> pooled fallback instead of the flat JSONL summary.
Dockerfile and the GitHub Actions daily workflow are in place, pending a
`TIINGO_API_KEY` repo secret and a first successful scheduled run before the
Claude Code Routine is retired.

## Why this exists

The end goal is a reference tool for swing trading: given a symbol and its
*current* SMC regime (choppy / trending / parabolic, up or down), look up
which strategy has historically performed best in that regime, and use that
to inform a trade. Not live, not for scalping or 0-2 DTE options -- the
daily-timeframe regime is the actual decision signal; hourly data is a
secondary layer for refining entry/exit timing within a swing thesis already
formed from the daily read, not a standalone signal.

## Decisions already made (don't re-litigate these without new information)

- **Storage: SQLite, not a vector database.** The core query is an exact
  structured lookup (ticker + regime + direction + interval -> strategy
  stats), not a similarity-search problem. A vector DB only becomes relevant
  later, if we ever want "find tickers whose regime history resembles this
  one" -- not needed for v1.
- **Granularity: pooled stats by default, symbol-specific override once a
  ticker has enough trades** (threshold ~15-20) to trust its own number over
  the pooled one. Per-symbol samples are too thin on their own for most
  regime/direction/strategy combinations at current data volumes.
- **Dashboard: a Claude Artifact, refreshed on request -- not a live/hosted
  site.** Artifacts can't read live data from a private git repo directly
  (sandboxed, no filesystem/network access to it), so "live" would require
  either a publicly-hosted static site (GitHub Pages is free but makes the
  site public by default on free-tier repos -- a real concern for trading
  data) or a hosted API with auth. Given live isn't needed for swing trading,
  the simplest and most private option wins: pull latest data, bake into an
  artifact, publish/republish when asked. Revisit only if "always current
  without asking Claude" becomes a real requirement.
- **Cloud runner: GitHub Actions (default recommendation) vs. Azure
  Container Apps Jobs -- pending the user's account situation.** Either way:
  add a `Dockerfile`, and the job still just computes stats and commits the
  updated SQLite file back to git -- the storage plan doesn't change based on
  which cloud executes it. Azure Container Apps Jobs is the clean fit if
  Azure is chosen (scheduled container, no VM to manage); GitHub Actions is
  simplest if staying on GitHub (same repo, ~10 min/day is comfortably inside
  free-tier minutes).
- **The current daily Routine (Claude Code trigger) is a stopgap, not the
  final architecture.** It already failed silently once (see git log
  2026-08-17/18 commits around `18d7c5a`) because a trigger-fired session's
  execution environment isn't fully visible/debuggable. Moving to GitHub
  Actions or Azure removes that fragility -- this is a real motivation for
  the migration, not just a preference.
- **New strategies arrive as Pine Script from the user**, translated into
  Python and registered in `smc_regime/strategies.py`'s `STRATEGIES` dict.
  Most Pine indicator logic maps directly onto what's already in
  `indicators.py`; flag anything that doesn't port cleanly (trailing stops,
  pyramiding, etc. that the current single-position long-only engine doesn't
  model) rather than silently approximating it.
- **TradingView Remix MCP (`TradingView_Remix`, aka tvremix) is the intended
  live-chart-context bridge** -- eventually, "what symbol/timeframe is the
  user looking at right now" comes from its `my_charts` tool, so a
  recommendation can be given without the user typing the ticker. Connection
  status is intermittent/unverified as of this writing (the server has
  connected and disconnected multiple times this session) -- verify before
  building on it.
- **Sector lists already exist** (`smc_regime/sectors/*.txt`, all 11 GICS
  sectors, 503 S&P 500 tickers) for running the same analysis scoped to a
  sector later.

## Build order, when the user says go

1. **Done.** SQLite trade-level store: `smc_regime/db.py`, schema
   `trades(run_at, interval, ticker, strategy, regime, direction, entry_date,
   exit_date, return_pct, win)` -- one row per trade, not pre-pooled, so
   pooling happens at query time -- plus `ticker_metadata(ticker, exchange,
   sector, industry, last_updated)`. Sector resolution
   (`smc_regime/metadata.py`) combines the scraped GICS sector files (S&P 500
   coverage) with a hand-mapped `SECTOR_OVERRIDES` table for the 22
   non-S&P names in the tracking universe (ETFs, ADRs, recent IPOs) --
   all 132 tracked tickers resolve to a real sector, none fall through to
   "Unknown". Exchange comes from Tiingo's `/tiingo/daily/{ticker}` metadata
   endpoint, cached to `smc_regime/data/exchange_cache.json`.
2. **Done.** `Dockerfile` wraps `smc_regime` + deps, entrypoint
   `python -m smc_regime.daily_snapshot`.
3. **In progress.** `.github/workflows/daily-snapshot.yml` runs the snapshot
   weekdays at 21:00 UTC and commits `backtest_logs/` back to the repo.
   Needs: `TIINGO_API_KEY` added as a repo secret, then a verified successful
   scheduled (or manually dispatched) run before retiring the Claude Code
   Routine (`trig_01S1Fvh2utK5MZhGVm7XvVcs`).
4. Verify the TradingView Remix connection and what `my_charts` actually
   returns.
5. **Done.** `recommend_cli` queries SQLite via `db.best_strategy()` with the
   symbol -> sector -> pooled fallback (min 15 trades per tier).
6. Build the first Artifact dashboard from the SQLite data, daily-regime-
   first layout with hourly as supporting detail.

## What NOT to do without asking

- Don't build a live/hosted dashboard (public GitHub Pages, etc.) without
  explicitly re-confirming the privacy tradeoff with the user.
- Don't add a vector database unless a real similarity-search use case shows
  up (not just "more data" -- pooled/symbol-specific SQLite handles scale
  fine on its own).
- Don't approximate Pine Script strategy behavior the current engine can't
  model (trailing stops, pyramiding, shorting) -- flag it instead.
