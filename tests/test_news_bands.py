"""The news bands are calibrated to one corpus, so they can go stale silently.

Two failures are possible and neither raises: a band nobody can reach (the
first live run had exactly that -- "very negative" needed a score below -0.50
when the minimum observed was -0.494), and the dashboard banding drifting away
from the module's after a recalibration, because the label the nightly stored
last night was written by the OLD thresholds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime.news import _BANDS, _FLOOR_LABEL, _label
from smc_regime.setup_score import _news_label

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


LABELS = [name for _, name in _BANDS] + [_FLOOR_LABEL]

# The observed spread of the feed these bands describe, from the first full
# run carrying real headlines: 408 tickers, median +0.344.
CORPUS_MIN, CORPUS_MAX = -0.494, 0.917

check(
    "1. every band is reachable inside the observed corpus range",
    all(
        any(
            _label(CORPUS_MIN + (CORPUS_MAX - CORPUS_MIN) * i / 2000) == name
            for i in range(2001)
        )
        for name in LABELS
    ),
)

# Thresholds are inclusive lower bounds, checked one step either side so a
# transposed comparison or a reordered list shows up.
step = 1e-9
check(
    "2. each threshold is an inclusive lower bound",
    all(
        _label(t) == name and _label(t - step) != name
        for t, name in _BANDS
    ),
)

check(
    "3. bands are listed high to low, so the scan returns the right one",
    [t for t, _ in _BANDS] == sorted((t for t, _ in _BANDS), reverse=True),
)

check(
    "4. the extremes land in the outer bands",
    _label(1.0) == _BANDS[0][1] and _label(-1.0) == _FLOOR_LABEL,
)

# The dashboard re-bands the stored NUMBER. A row written under the old
# thresholds must come back with today's label, not last night's.
check(
    "5. the dashboard label is derived from the score, not the stored text",
    all(_news_label(t) == _label(t) for t, _ in _BANDS),
)

check(
    "6. a missing score reads as no data rather than a band",
    _news_label(None) == "no data" and _news_label(float("nan")) == "no data",
)

# Only the ROUNDED score is stored, so the label written beside it must be a
# function of that rounded value. Labelling the full-precision mean instead
# let the stored label disagree with what the stored number re-derives to,
# for any ticker whose mean rounds across a threshold -- LEN, FLEX and LVS
# on run #29. Driven through the real ticker_sentiment() with the fetch and
# the scorer stubbed, so it covers the code path that actually writes the row.
import smc_regime.news as news_mod

_CUT = _BANDS[1][0]                       # 0.52, the "stronger" cut
_FAKE = [0.519, 0.519, 0.521]             # mean 0.51966... -> rounds to 0.52
_real_fetch, _real_score = news_mod.fetch_news, news_mod.score_headline
try:
    news_mod.fetch_news = lambda ticker, days=7, limit=20: [
        {"title": f"h{i}"} for i in range(len(_FAKE))
    ]
    _seq = iter(_FAKE)
    news_mod.score_headline = lambda text: next(_seq)
    result = news_mod.ticker_sentiment("TEST")
finally:
    news_mod.fetch_news, news_mod.score_headline = _real_fetch, _real_score

check(
    "7. the mean is stored rounded and lands on the threshold",
    result["avg_compound"] == _CUT and sum(_FAKE) / len(_FAKE) < _CUT,
)

check(
    "8. the stored label matches what the stored score re-derives to",
    result["label"] == _label(result["avg_compound"]) == _BANDS[1][1],
)

check(
    "9. per-article labels follow the same rule as the average",
    all(a["label"] == _label(a["compound"]) for a in result["articles"]),
)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
