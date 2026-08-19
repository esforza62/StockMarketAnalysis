"""Download the latest SQLite snapshot from the `data-latest` GitHub Release.

The trade-level store (backtest_logs/smc_regime.db) is rewritten in full
every night, and SQLite files don't diff well -- committing it directly to
git would grow the repository's history by roughly one full copy of the
file per day, forever, and risks hitting GitHub's 100MB per-file hard
limit as more strategies/tickers get added. Publishing it as a release
asset instead keeps it out of git history entirely, with no size limit
that matters here (up to 2GB per asset) and no metered storage/bandwidth
cost, at the price of needing an explicit download step after cloning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from . import db as db_module

RELEASE_TAG = "data-latest"
ASSET_NAME = "smc_regime.db"
_ASSET_URL = f"https://github.com/esforza62/StockMarketAnalysis/releases/download/{RELEASE_TAG}/{ASSET_NAME}"


def fetch(dest: str | Path = db_module.DEFAULT_DB_PATH, force: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"{dest} already exists, skipping (pass --force to re-download)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {_ASSET_URL} -> {dest}")
    response = requests.get(_ASSET_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Downloaded {len(response.content) / 1e6:.1f} MB")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the latest SQLite snapshot from GitHub Releases.")
    parser.add_argument("--dest", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = parser.parse_args()

    try:
        fetch(args.dest, force=args.force)
    except requests.HTTPError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("No release published yet, or the asset name/tag changed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
