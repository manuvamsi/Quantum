#!/usr/bin/env bash
# Launch the ReQAgnIze edge appliance (kiosk).
cd "$(dirname "$0")/.."

# Force display for Wayland/Xwayland (SSH sessions)
if [ -z "$DISPLAY" ] && [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
    echo "[run.sh] Set DISPLAY=:0"
fi
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m app.main
