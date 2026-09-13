#!/bin/bash
# ReQAgnIze Launcher — sets display environment for Wayland/Xwayland
# Run this instead of 'python -m app.main'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect display
if [ -z "$DISPLAY" ]; then
    # Check for Xwayland :0
    if [ -S /tmp/.X11-unix/X0 ]; then
        export DISPLAY=:0
        echo "[launcher] Set DISPLAY=:0 (Xwayland)"
    else
        echo "[launcher] WARNING: No X11 display found!"
        echo "  Run from the Jetson desktop, or set DISPLAY manually."
    fi
fi

# Use xcb platform for Qt (works with Xwayland)
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

# Activate venv
source .venv/bin/activate

echo "[launcher] DISPLAY=$DISPLAY"
echo "[launcher] QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "[launcher] Starting ReQAgnIze..."
echo ""

exec python -m app.main "$@"
