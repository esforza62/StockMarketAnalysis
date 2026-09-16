"""The dashboards are versioned as templates; this pins the bake.

Both pages were once un-versioned: the only copies were a scratchpad file
and the published artifact, and a reclaimed container took the scratchpad
with it. These checks cover the two ways the template split could rot --
a template that no longer accepts a payload, and a bake that silently
does nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smc_regime import build_dashboard

failures = []


def check(label, ok):
    print(f"{label}  {'OK' if ok else 'FAILED'}")
    if not ok:
        failures.append(label)


check(
    "1. both dashboards are registered",
    sorted(build_dashboard.DASHBOARDS) == ["regime_desk", "setup_grades"],
)

check(
    "2. every registered dashboard has a committed template",
    all(build_dashboard.template_path(n).is_file() for n in build_dashboard.DASHBOARDS),
)

# The templates carry an EMPTY payload. That is what keeps them small
# enough to diff -- a built page is ~800KB of snapshot.
for name in build_dashboard.DASHBOARDS:
    size = build_dashboard.template_path(name).stat().st_size
    check(f"3. {name} template is markup-sized, not snapshot-sized ({size//1024}KB)", size < 200_000)

# Each template's placeholder must parse, so the committed file is a
# working page rather than one that throws the moment it is opened.
for name, spec in build_dashboard.DASHBOARDS.items():
    html = build_dashboard.template_path(name).read_text()
    marker = f'<script id="{spec["script_id"]}" type="application/json">\n'
    start = html.index(marker) + len(marker)
    end = html.index("\n</script>", start)
    try:
        placeholder = json.loads(html[start:end])
        ok = isinstance(placeholder, dict)
    except Exception:
        ok = False
    check(f"4. {name} template placeholder is valid JSON", ok)

# A payload bakes in, and comes back out unchanged.
sample = {
    "run_at": "2026-09-16T00:00:00+00:00", "interval": "1d", "min_trades": 15,
    "ticker_count": 1, "technicals_covered": 1, "news_covered": 1,
    "grade_counts": {"A": 1}, "macro": {"levels": [], "events": []},
    "sectors": ["Energy"], "tickers": [{"ticker": "XOM", "grade": "A"}],
}
built = build_dashboard.build("setup_grades", sample)
marker = '<script id="setup-score-data" type="application/json">\n'
start = built.index(marker) + len(marker)
round_tripped = json.loads(built[start:built.index("\n</script>", start)])

check("5. the baked payload round-trips exactly", round_tripped == sample)

# The id appears twice by design -- once on the block, once in the
# getElementById that reads it -- so count the TAG, not the string.
check(
    "6. baking leaves the surrounding markup alone",
    "<title>Setup Grades</title>" in built
    and built.count('<script id="setup-score-data"') == 1
    and "getElementById('setup-score-data')" in built,
)

# Baking the placeholder back in changes nothing, and that must RAISE
# rather than pass: a swap that quietly did not happen is how a page went
# out carrying a three-hour-old timestamp while the tooling said OK.
tpl = build_dashboard.template_path("setup_grades").read_text()
start = tpl.index(marker) + len(marker)
placeholder = json.loads(tpl[start:tpl.index("\n</script>", start)])
try:
    build_dashboard.build("setup_grades", placeholder)
    raised = False
except ValueError:
    raised = True
check("7. a bake that changes nothing raises instead of reporting success", raised)

# A template missing its block must fail loudly too.
try:
    build_dashboard.build("setup_grades", sample, template="<html>no block here</html>")
    raised = False
except ValueError:
    raised = True
check("8. a template with no payload block raises", raised)

try:
    build_dashboard.template_path("nope")
    raised = False
except KeyError:
    raised = True
check("9. an unknown dashboard name raises", raised)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
