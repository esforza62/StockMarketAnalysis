"""Put the append-only snapshot log back in order after a merge or rebase.

`backtest_logs/regime_strategy_log.jsonl` gets one record appended per
(run, interval). When two snapshot runs overlap -- a dispatched run still
backtesting while the nightly one pushes -- both append after the same
ancestor, and a plain rebase conflicts on every appended line. The workflow
step runs under `bash -e`, so that conflict used to kill the push outright
and silently drop a run's records.

`merge=union` in .gitattributes fixes the failure by keeping both sides'
lines, but it concatenates them in whatever order the merge produced and
would happily keep a duplicate. This restores the file's two invariants
afterwards: one record per (run_at, interval), ordered by run_at.

Dedup keeps the LAST record for a key. Re-running a snapshot for an
interval is a correction, so the later write is the one to trust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize(path: Path) -> tuple[int, int]:
    """Rewrite `path` deduped on (run_at, interval) and sorted by run_at.

    Returns (records_kept, duplicates_dropped). Leaves the file untouched
    when it is already normalized, so the caller can tell whether there is
    anything to commit.
    """
    original = path.read_text()
    lines = [line for line in original.splitlines() if line.strip()]

    # Only reachable if the union merge driver did not apply (a checkout
    # whose .gitattributes predates it, say). Bail out loudly rather than
    # letting a raw decode error suggest the log itself is corrupt.
    markers = [ln for ln in lines if ln.startswith(("<<<<<<<", "=======", ">>>>>>>"))]
    if markers:
        raise ValueError(
            f"{path} still has {len(markers)} conflict marker line(s) -- the union merge "
            "driver did not apply. Check that .gitattributes is present in the tree being "
            "merged into, then resolve the conflict before normalizing."
        )

    records = [json.loads(line) for line in lines]

    deduped: dict[tuple[str, str], dict] = {}
    for record in records:
        deduped[(record["run_at"], record["interval"])] = record

    ordered = sorted(deduped.values(), key=lambda r: (r["run_at"], r["interval"]))
    rebuilt = "".join(json.dumps(r) + "\n" for r in ordered)

    if rebuilt != original:
        path.write_text(rebuilt)
    return len(ordered), len(records) - len(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="backtest_logs/regime_strategy_log.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"{path}: not found, nothing to normalize", file=sys.stderr)
        return

    kept, dropped = normalize(path)
    print(f"{path}: {kept} records" + (f", dropped {dropped} duplicate(s)" if dropped else ""))


if __name__ == "__main__":
    main()
