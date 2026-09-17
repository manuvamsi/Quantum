#!/usr/bin/env bash
# Launch the ReQAgnIze edge appliance (kiosk).
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m app.main
