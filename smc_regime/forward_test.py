"""Out-of-sample test of the regime desk's strategy picks.

WHAT THIS EXISTS TO FIX. regime_strategy_log.jsonl ranks strategies on a
rolling multi-year window that CONTAINS the trades it is ranking, so the
ranking and its evaluation share the same data. The 30-day consistency
check found the top pick unchanged on 17 of 21 buckets with a 97% modal
share -- but consecutive runs differ by a median 0.07% of their trades, so
a stable ranking over near-identical data is close to arithmetic rather
than evidence. It measures reproducibility, not skill.

The fix is temporal separation: take the pick FROZEN at date T from the
log, and score it only on trades ENTERED after T. The log is what makes
this possible -- each record carries run_at and the full ranking as of
that date, so it is a pre-registration nobody can retroactively edit.

THE CENSORING PROBLEM, which dominates everything here. run_backtest()
appends a trade only when it closes; a position still open when the bars
run out is dropped silently, with no final mark-to-market. So `trades`
holds closed trades only, and a strategy is missing its recent entries in
proportion to how long it holds. Measured against each strategy's own
historical entry rate on the 1d run, the final hold-window is missing
~72% of expected entries for rsi_dip_recovery (240-day holds) and ~88%
for rsi_dip_trend_filter (250 days), against ~0% for the 5-12 day
strategies.

That is fatal to a naive forward test. Filtering entry_date > T and
comparing mean returns would score short-hold strategies on a nearly
complete sample and long-hold ones on whichever few trades happened to
close early -- and for a dip-and-recovery strategy the trades that close
early are the ones that recovered. The comparison would not be noisy, it
would be backwards.

So coverage is computed per strategy and reported beside every figure,
and a comparison whose two sides have materially different coverage is
labelled UNRELIABLE rather than printed as a number. Refusing to answer
is the correct output here; the alternative is a decisive-looking result
that measures hold length.

NOTE the coverage estimate is a CEILING on what can be trusted, not a
correction. It models trades censored by the end of the window. It cannot
see positions that never close at all -- a strategy with no time stop
holds a losing trade forever, and those are absent from every point in
the history, not just the recent tail. Bounding that needs a re-run with
a forced close-out at the last bar, which needs price bars.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import db as db_module

DEFAULT_LOG = Path("backtest_logs/regime_strategy_log.jsonl")
DEFAULT_MIN_TRADES = 15

# Two strategies whose observable fractions differ by more than this are
# not measuring the same thing, so their difference is not reported as a
# result. 0.15 is deliberately strict: the whole failure mode here is a
# hold-length artifact wearing the costume of an edge.
COVERAGE_TOLERANCE = 0.15

# Below this, a strategy's forward sample is so truncated that its mean is
# not a description of the strategy at all.
MIN_COVERAGE = 0.25


def load_picks(log_path: Path, as_of: str, min_trades: int = DEFAULT_MIN_TRADES) -> dict:
    """The rankings frozen in the log on the last run at or before `as_of`.

    Returns {(interval, regime, direction): {...}} carrying the pick, the
    runner-up margin and the median-ranked strategy -- everything the
    benchmarks below need, all of it decided by data the log had at the
    time.
    """
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda r: r["run_at"])

    latest: dict[str, dict] = {}
    for record in records:
        if record["run_at"][:10] <= as_of:
            latest[record["interval"]] = record

    picks: dict[tuple[str, str, str], dict] = {}
    for interval, record in latest.items():
        buckets: dict[tuple[str, str], list] = defaultdict(list)
        for entry in record["summary"]:
            buckets[(entry["regime"], entry["direction"])].append(entry)
        for (regime, direction), entries in buckets.items():
            eligible = sorted(
                (e for e in entries if e.get("trade_count", 0) >= min_trades),
                key=lambda e: -e["avg_return_pct"],
            )
            if not eligible:
                continue
            picks[(interval, regime, direction)] = {
                "as_of": record["run_at"][:10],
                "pick": eligible[0]["strategy"],
                "pick_in_sample_return": eligible[0]["avg_return_pct"],
                # A margin near zero means the pick is a tie-break, not a
                # preference: 1h trending/down separates its top two by
                # 0.11pp and 15m parabolic/up by 0.06pp.
                "margin": (eligible[0]["avg_return_pct"] - eligible[1]["avg_return_pct"])
                if len(eligible) > 1 else None,
                "median_rank": eligible[len(eligible) // 2]["strategy"],
                "ranked": [e["strategy"] for e in eligible],
            }
    return picks


def _run_ids(conn) -> dict[str, int]:
    return {interval: rid for rid, interval in conn.execute("SELECT id, interval FROM runs")}


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def forward_trades(conn, run_id: int, regime: str, direction: str, after: str) -> dict[str, list[float]]:
    """Returns per strategy, after `after`, in this bucket."""
    rows = conn.execute(
        """SELECT s.name, t.return_pct
             FROM trades t JOIN strategies s ON s.id = t.strategy_id
            WHERE t.run_id = ? AND t.regime = ? AND t.direction = ? AND t.entry_date > ?""",
        (run_id, regime, direction, _epoch(after)),
    ).fetchall()
    out: dict[str, list[float]] = defaultdict(list)
    for name, ret in rows:
        out[name].append(ret)
    return out


# Average hold and window end depend only on the run, not on the bucket, so
# they are read once per run rather than re-scanning several million trade
# rows for every (bucket, strategy) pair -- which is the difference between
# this finishing in a second and in several minutes.
_HOLD_CACHE: dict[int, dict[str, float]] = {}
_END_CACHE: dict[int, int] = {}


def _holds(conn, run_id: int) -> dict[str, float]:
    if run_id not in _HOLD_CACHE:
        _HOLD_CACHE[run_id] = {
            name: (hold or 0.0)
            for name, hold in conn.execute(
                """SELECT s.name, AVG((t.exit_date - t.entry_date) / 86400.0)
                     FROM trades t JOIN strategies s ON s.id = t.strategy_id
                    WHERE t.run_id = ? GROUP BY s.name""",
                (run_id,),
            )
        }
    return _HOLD_CACHE[run_id]


def _window_end(conn, run_id: int) -> int:
    if run_id not in _END_CACHE:
        _END_CACHE[run_id] = conn.execute(
            "SELECT MAX(entry_date) FROM trades WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    return _END_CACHE[run_id]


def coverage(conn, run_id: int, strategy: str, after: str) -> float:
    """Fraction of the forward window in which this strategy's trades could
    have closed and therefore been recorded at all.

    A strategy holding H days cannot show a trade entered in the final H
    days of the window, because run_backtest never writes an open
    position. With a window of W days the observable share is
    (W - H) / W, floored at zero. At W=24 and H=240 that is 0 -- which is
    the honest answer for a daily dip-recovery pick three weeks out, and
    the reason this function exists rather than reporting a mean over
    whatever happened to close.
    """
    hold_days = _holds(conn, run_id).get(strategy, 0.0)
    window_days = (_window_end(conn, run_id) - _epoch(after)) / 86400.0
    if window_days <= 0:
        return 0.0
    return max(0.0, (window_days - hold_days) / window_days)


def _describe(returns: list[float]) -> dict:
    if not returns:
        return {"n": 0, "mean": None, "median": None, "win_rate": None}
    return {
        "n": len(returns),
        "mean": statistics.fmean(returns),
        "median": statistics.median(returns),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) * 100,
    }


def evaluate_bucket(conn, run_id: int, interval: str, regime: str, direction: str,
                    pick_info: dict, after: str) -> dict:
    """Score the frozen pick against the benchmarks, with coverage attached."""
    by_strategy = forward_trades(conn, run_id, regime, direction, after)
    pick = pick_info["pick"]

    pick_stats = _describe(by_strategy.get(pick, []))
    pick_cov = coverage(conn, run_id, pick, after)

    # Benchmark 1: equal-weight across every strategy that traded this
    # bucket -- the no-skill baseline. If the pick cannot beat this, the
    # ranking is not adding anything.
    pooled = [r for rets in by_strategy.values() for r in rets]
    equal_weight = _describe(pooled)

    # Benchmark 2: the median-ranked strategy, which separates "being top"
    # from merely "being in the list".
    median_name = pick_info["median_rank"]
    median_stats = _describe(by_strategy.get(median_name, []))
    median_cov = coverage(conn, run_id, median_name, after)

    # Mean coverage of the pooled benchmark, weighted by each strategy's
    # contribution -- an equal-weight mean over mostly-short-hold
    # strategies is a mostly-complete sample, and comparing a censored
    # pick against it is the trap this whole module is built around.
    if pooled:
        weighted = sum(coverage(conn, run_id, s, after) * len(r) for s, r in by_strategy.items())
        pooled_cov = weighted / len(pooled)
    else:
        pooled_cov = 0.0

    def verdict(stats, cov, label):
        if pick_stats["n"] == 0:
            return None, f"no forward trades for {pick} (coverage {pick_cov:.0%})"
        if stats["n"] == 0:
            return None, f"no forward trades for {label}"
        if pick_cov < MIN_COVERAGE:
            return None, f"UNRELIABLE: {pick} coverage {pick_cov:.0%} < {MIN_COVERAGE:.0%}"
        if abs(pick_cov - cov) > COVERAGE_TOLERANCE:
            return None, (f"UNRELIABLE: coverage mismatch "
                          f"({pick} {pick_cov:.0%} vs {label} {cov:.0%})")
        return pick_stats["mean"] - stats["mean"], "ok"

    excess_ew, note_ew = verdict(equal_weight, pooled_cov, "equal-weight")
    excess_md, note_md = verdict(median_stats, median_cov, median_name)

    return {
        "interval": interval, "regime": regime, "direction": direction,
        "pick": pick, "pick_as_of": pick_info["as_of"],
        "in_sample_margin": pick_info["margin"],
        "pick_forward": pick_stats, "pick_coverage": pick_cov,
        "equal_weight": equal_weight, "equal_weight_coverage": pooled_cov,
        "median_rank": median_name, "median_forward": median_stats, "median_coverage": median_cov,
        "excess_vs_equal_weight": excess_ew, "note_equal_weight": note_ew,
        "excess_vs_median": excess_md, "note_median": note_md,
    }


def run(as_of: str, log_path: Path = DEFAULT_LOG, db_path=None,
        intervals: list[str] | None = None, min_trades: int = DEFAULT_MIN_TRADES) -> list[dict]:
    picks = load_picks(log_path, as_of, min_trades)
    conn = db_module.connect(db_path) if db_path else db_module.connect()
    try:
        ids = _run_ids(conn)
        results = []
        for (interval, regime, direction), info in sorted(picks.items()):
            if intervals and interval not in intervals:
                continue
            if interval not in ids:
                continue
            results.append(
                evaluate_bucket(conn, ids[interval], interval, regime, direction, info, as_of)
            )
        return results
    finally:
        conn.close()


def _format(results: list[dict]) -> str:
    lines = []
    head = (f"{'interval':<6}{'bucket':<20}{'pick':<24}{'fwd n':>7}{'cov':>6}"
            f"{'pick ret':>10}{'bench ret':>11}{'excess':>9}  note")
    lines.append(head)
    lines.append("-" * len(head))
    reportable = 0
    for r in results:
        pf, ew = r["pick_forward"], r["equal_weight"]
        excess = r["excess_vs_equal_weight"]
        if excess is not None:
            reportable += 1
        pick_ret = "-" if pf["mean"] is None else format(pf["mean"], ".2f")
        bench_ret = "-" if ew["mean"] is None else format(ew["mean"], ".2f")
        exc = "-" if excess is None else format(excess, "+.2f")
        note = "" if r["note_equal_weight"] == "ok" else r["note_equal_weight"]
        bucket = r["regime"] + "/" + r["direction"]
        lines.append(
            f"{r['interval']:<6}{bucket:<20}{r['pick']:<24}"
            f"{pf['n']:>7}{r['pick_coverage']:>5.0%}"
            f"{pick_ret:>10}{bench_ret:>11}{exc:>9}  {note}"
        )
    lines.append("")
    lines.append(f"reportable buckets: {reportable}/{len(results)}"
                 f"  (the rest are censored by hold length, not by a lack of edge)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--as-of", required=True, help="pick date, YYYY-MM-DD; picks are read from the log as frozen on that date")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--db-file", default=None)
    parser.add_argument("--intervals", nargs="*", default=None)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--json", action="store_true", help="emit raw results")
    args = parser.parse_args()

    results = run(args.as_of, Path(args.log_file), args.db_file, args.intervals, args.min_trades)
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(_format(results))


if __name__ == "__main__":
    main()
