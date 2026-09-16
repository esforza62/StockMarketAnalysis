"""Bake an exported payload into a dashboard template.

THE POINT OF THIS MODULE is that the dashboards are now versioned as
TEMPLATES -- markup with an empty payload -- and the publishable page is
built from a template plus a fresh export. Before this existed the only
copies of the built pages lived in a scratchpad and in the published
artifacts, and a reclaimed container took the scratchpad with it: both
pages had to be recovered by reading them back out of the artifact
service, which only worked because they happened to still be published.

The split is deliberate rather than incidental:

  * The TEMPLATE is markup and belongs in git, where changes to it show up
    as reviewable diffs. It is ~60KB.
  * The PAYLOAD is a snapshot of a 340MB database and does not, which is
    also why a built page is ~800KB and would land as an unreadable
    800KB diff every single night.

A template is a working page on its own -- its payload is empty but
structurally complete, so opening one gives the real chrome in its empty
state. That is what you want when editing markup with no database around,
and it means a broken template fails when you open it rather than during
a publish.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import jsonfmt

TEMPLATE_DIR = Path(__file__).parent / "dashboards"

# Which JSON block each dashboard's payload lives in, and the nesting depth
# that puts one record per line. The depth must match what the exporter
# uses or the rebuilt file would differ from a directly-exported one by
# nothing but whitespace -- see tests/test_build_dashboard.py, which pins
# exactly that.
DASHBOARDS = {
    "setup_grades": {
        "script_id": "setup-score-data",
        "compact_depth": 2,
        "exporter": "smc_regime.export_setup_score_data",
    },
    "regime_desk": {
        "script_id": "dashboard-data",
        "compact_depth": 5,
        "exporter": "smc_regime.export_dashboard_data",
    },
}


def template_path(name: str) -> Path:
    if name not in DASHBOARDS:
        raise KeyError(f"unknown dashboard {name!r}; expected one of {sorted(DASHBOARDS)}")
    return TEMPLATE_DIR / f"{name}.html"


def build(name: str, payload: dict, template: str | None = None) -> str:
    """Return the template for `name` with `payload` baked into it.

    Raises rather than returning the template untouched when the block is
    missing or the substitution changes nothing: a swap that quietly did
    not happen is how a page went out with a three-hour-old timestamp
    while the tooling reported success.
    """
    spec = DASHBOARDS[name]
    html = template if template is not None else template_path(name).read_text()
    baked = jsonfmt.dumps(payload, compact_depth=spec["compact_depth"])

    pattern = re.compile(
        r'(<script id="%s" type="application/json">\n)(.*?)(\n</script>)'
        % re.escape(spec["script_id"]),
        re.DOTALL,
    )
    out, n = pattern.subn(lambda m: m.group(1) + baked + m.group(3), html, count=1)
    if n != 1:
        raise ValueError(
            f"{name}: expected exactly one <script id=\"{spec['script_id']}\"> block, found {n}"
        )
    if out == html:
        raise ValueError(
            f"{name}: baking changed nothing -- the payload is identical to the "
            "template's placeholder, which almost certainly means a stale or "
            "empty payload file"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dashboard", choices=sorted(DASHBOARDS))
    parser.add_argument("--data", help="exported payload JSON; omit to export fresh")
    parser.add_argument("--out", required=True, help="write the built page here")
    parser.add_argument("--db-file", default=None, help="passed through when exporting fresh")
    args = parser.parse_args()

    if args.data:
        payload = json.loads(Path(args.data).read_text())
    else:
        import subprocess
        cmd = [sys.executable, "-m", DASHBOARDS[args.dashboard]["exporter"]]
        if args.db_file:
            cmd += ["--db-file", args.db_file]
        payload = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)

    out = Path(args.out)
    out.write_text(build(args.dashboard, payload))
    print(f"built {args.dashboard} -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
