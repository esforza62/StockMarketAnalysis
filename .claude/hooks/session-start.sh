#!/bin/bash
# Installs what a Claude Code on the web session needs to run this repo.
# Local sessions are left alone -- developers manage their own venvs.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# smc_regime is imported as a top-level package from the repo root.
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"

# pip install (not a locked sync) so the cached container layer is reused
# on later sessions instead of rebuilding from scratch.
python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt

# Everything in smc_regime that touches Tiingo reads TIINGO_API_KEY from the
# environment. It is deliberately NOT set here -- it is a secret and this file
# is tracked. Set it in the environment's settings on claude.ai so it is
# present in the session; without it, price/news/metadata fetches raise.
if [ -z "${TIINGO_API_KEY:-}" ]; then
  echo "session-start: TIINGO_API_KEY is not set -- Tiingo fetches will fail." >&2
fi
