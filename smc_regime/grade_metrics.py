"""Is the grading still working? A tracked metric, not a one-off verdict.

`grade_report` answers "did the grades work" for the history as it stands
today. That is the right question to ask once. It is the wrong thing to
watch, for two reasons this module exists to fix.

THE ORDERING FLAG IS TOO BRITTLE TO TRACK. It asks whether mean forward
return falls monotonically A -> B -> C -> D, which one noisy bucket flips.
On the first history deep enough to judge, the 1-month horizon read BROKEN
on a single A-vs-B inversion over an 89-row A bucket -- while its rank
correlation was +0.103 and positive on 5 of 5 capture dates, as strong as
the 2-week horizon that read "holds". A metric that says BROKEN when the
signal is intact will cry wolf until nobody reads it.

A POOLED NUMBER CANNOT SHOW DEGRADATION. Recomputing over all history each
night mixes every regime together, so a grade that stopped working last
month is diluted by the two months before it. The failure this is meant to
catch -- the scorer quietly ceasing to rank -- looks like a slowly falling
number, and you cannot see a trend in a single pooled figure.

So the primary metric here is the INFORMATION COEFFICIENT: the Spearman
rank correlation between a row's score and its forward return, computed
PER CAPTURE DATE and then summarised. One IC per date gives a series with a
mean, a dispersion and a hit rate (how often it is positive at all), which
is both more robust than the binary flag and able to show a trend.

It is computed on `total_points`, not on the letter. The raw score carried
more signal at every horizon measured (+0.039/+0.113/+0.103 against
+0.036/+0.113/+0.089 for the letter), and more importantly the two answer
different questions: the score IC says whether the SIGNAL ranks, while the
grade distribution says whether the CUTS still map it sensibly. Those fail
independently -- cuts drift while the signal is fine, or the signal decays
while the distribution looks healthy -- and a metric that blends them tells
you something is wrong without telling you which thing.

Scale: a cross-sectional IC around 0.10 is a strong single signal; 0.02-0.05
is typical and still useful. Near zero means the score is not ranking. What
matters more than the level is the TREND and the hit rate.

    python -m smc_regime.grade_metrics            # current, vs recorded history
    python -m smc_regime.grade_metrics --record   # append today's row
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import grade_history as gh
from . import grade_report as gr

_METRICS_PATH = Path(__file__).parent / "data" / "grade_metrics.jsonl"
_GRADES = ("A", "B", "C", "D")
#: Dates with fewer rows than this are skipped: an IC over a handful of
#: names is noise, and averaging it in would widen the band for nothing.
_MIN_ROWS_PER_DATE = 30


def spearman(a, b) -> float:
    """Rank correlation. Pearson on ranks, so scipy stays off the dependency
    list -- it is not installed in the nightly's environment and adding it
    to buy one function would be a poor trade."""
    a, b = pd.Series(list(a)).rank(), pd.Series(list(b)).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b))


def information_coefficient(scored: pd.DataFrame, horizon: str,
                            score_col: str = "total_points") -> dict:
    """Per-date IC for one horizon, plus its summary.

    `hit_rate` is the share of dates with a POSITIVE IC. It is reported
    alongside the mean because they fail differently: a mean dragged
    positive by one extreme date is not the same as a signal that ranks
    correctly most days, and only the second is worth trusting.
    """
    col, mature = f"fwd_{horizon}", f"mature_{horizon}"
    if col not in scored.columns:
        return {}
    usable = scored[scored[mature] & scored[col].notna()]
    if usable.empty:
        return {"n": 0, "dates": 0, "pending": int((~scored[mature]).sum())}

    per_date = {}
    for as_of, g in usable.groupby("as_of"):
        if len(g) < _MIN_ROWS_PER_DATE:
            continue
        ic = spearman(g[score_col], g[col])
        if not np.isnan(ic):
            per_date[str(as_of)] = round(ic, 4)

    vals = np.array(list(per_date.values()), dtype=float)
    out = {
        "n": int(len(usable)),
        "dates": len(per_date),
        "ic_pooled": round(spearman(usable[score_col], usable[col]), 4),
        "ic_mean": round(float(vals.mean()), 4) if len(vals) else None,
        "ic_sd": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
        "ic_hit_rate": round(float((vals > 0).mean() * 100), 1) if len(vals) else None,
        "per_date": per_date,
    }

    # Economic magnitude alongside the rank statistic: an IC says the
    # ordering is right, not that acting on it would have paid.
    means = {g: usable[usable["grade"] == g][col] for g in _GRADES}
    out["bucket_n"] = {g: int(len(v)) for g, v in means.items()}
    out["spread_ad"] = (round(float(means["A"].mean() - means["D"].mean()), 3)
                        if len(means["A"]) and len(means["D"]) else None)
    out["winrate_gap_ad"] = (
        round(float((means["A"] > 0).mean() * 100 - (means["D"] > 0).mean() * 100), 1)
        if len(means["A"]) and len(means["D"]) else None)
    return out


def distribution(history: pd.DataFrame) -> dict:
    """Share of each grade on the most recent capture.

    Tracked separately from the IC on purpose. The cuts drifting and the
    signal decaying are different failures with different fixes, and a
    single combined health number would hide which one happened.
    """
    if history.empty:
        return {}
    latest = history[history["as_of"] == history["as_of"].max()]
    n = len(latest)
    return {
        "as_of": str(latest["as_of"].iloc[0]),
        "n": int(n),
        "shares_pct": {g: round(float((latest["grade"] == g).mean() * 100), 1) for g in _GRADES},
        "mean_score": round(float(latest["total_points"].mean()), 2),
        "score_p10": round(float(latest["total_points"].quantile(0.10)), 2),
        "score_p90": round(float(latest["total_points"].quantile(0.90)), 2),
    }


def compute(history: pd.DataFrame | None = None,
            horizons: dict[str, int] | None = None) -> dict:
    """Everything worth recording for one run."""
    if history is None:
        history = gh.load()
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    horizons = horizons or gr.DEFAULT_HORIZONS
    scored = gr.forward_returns(history, horizons)
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "history_rows": int(len(history)),
        "capture_dates": int(history["as_of"].nunique()) if len(history) else 0,
        "distribution": distribution(history),
        "horizons": {h: information_coefficient(scored, h) for h in horizons},
    }


def record(metrics: dict, path: Path = _METRICS_PATH) -> None:
    """Append one row. The per-date IC map is dropped from the stored row --
    it is recomputable from the history and would otherwise grow the file
    quadratically as every run re-records every past date."""
    row = json.loads(json.dumps(metrics))
    for h in row.get("horizons", {}).values():
        h.pop("per_date", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_recorded(path: Path = _METRICS_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record", action="store_true", help="append today's row to the tracked history")
    p.add_argument("--path", default=str(_METRICS_PATH))
    args = p.parse_args()

    m = compute()
    d = m["distribution"]
    print(f"{m['history_rows']} rows over {m['capture_dates']} capture dates")
    if d:
        sh = "  ".join(f"{g} {d['shares_pct'][g]:.1f}%" for g in _GRADES)
        print(f"latest capture {d['as_of']}: {sh}   mean score {d['mean_score']} "
              f"(p10 {d['score_p10']} / p90 {d['score_p90']})")

    print(f"\n{'horizon':<9}{'IC':>8}{'mean':>9}{'sd':>8}{'hit':>7}{'dates':>7}"
          f"{'n':>8}{'A-D':>8}{'winAD':>8}")
    for h, v in m["horizons"].items():
        if not v or not v.get("dates"):
            print(f"{h:<9}{'--':>8}   not enough matured rows yet")
            continue
        print(f"{h:<9}{v['ic_pooled']:>+8.4f}{v['ic_mean']:>+9.4f}"
              f"{(v['ic_sd'] if v['ic_sd'] is not None else float('nan')):>8.4f}"
              f"{v['ic_hit_rate']:>6.0f}%{v['dates']:>7}{v['n']:>8}"
              f"{v['spread_ad']:>+8.2f}{v['winrate_gap_ad']:>+7.1f}")

    hist = load_recorded(Path(args.path))
    if hist:
        print(f"\ntrend over {len(hist)} recorded run(s):")
        for h in m["horizons"]:
            series = [(r["computed_at"][:10], r["horizons"].get(h, {}).get("ic_mean"))
                      for r in hist if r["horizons"].get(h, {}).get("ic_mean") is not None]
            if series:
                print(f"  {h:<4} " + "  ".join(f"{d} {v:+.3f}" for d, v in series[-8:]))
    else:
        print("\nno recorded history yet -- run with --record to start the series")

    if args.record:
        record(m, Path(args.path))
        print(f"\nrecorded -> {args.path}")


if __name__ == "__main__":
    main()
