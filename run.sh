#!/usr/bin/env bash
set -euo pipefail
# Activate the project venv if present (setup: see project.md);
# otherwise fall back to the ambient environment.
VENV="$(cd "$(dirname "$0")" && pwd)/.venv"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
exec python -m ragdepot "$@"
