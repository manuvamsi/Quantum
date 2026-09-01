#!/usr/bin/env bash
# setup_jetson.sh — one-shot environment setup for the ReQAgnIze edge appliance.
# Run ON THE JETSON from the NVIDIA_App/ directory:  bash scripts/setup_jetson.sh
#
# It: detects L4T/JetPack, (optionally) maxes performance, creates a venv with
# system site-packages, and installs DepthAI, ChromaDB, PennyLane(+lightning),
# Jetson.GPIO, and the app deps. PyTorch on Jetson is JetPack-specific, so it is
# detected and guided rather than blindly installed.

set -euo pipefail
cd "$(dirname "$0")/.."   # NVIDIA_App/
ROOT="$(pwd)"
echo "== ReQAgnIze edge setup :: $ROOT =="

# ---- 1. Detect L4T / JetPack ------------------------------------------------
L4T="unknown"
if [ -f /etc/nv_tegra_release ]; then
  # e.g. "# R35 (release), REVISION: 4.1" -> 35.4
  L4T=$(sed -n 's/.*R\([0-9]\+\).*REVISION: \([0-9]\+\).*/\1.\2/p' /etc/nv_tegra_release | head -1)
fi
L4T_MAJOR="${L4T%%.*}"
case "$L4T_MAJOR" in
  32) JP="JetPack 4.x" ;;
  35) JP="JetPack 5.x" ;;
  36) JP="JetPack 6.x" ;;
  *)  JP="unknown (L4T=$L4T)" ;;
esac
echo "Detected L4T $L4T  ->  $JP"

# ---- 2. Optional: max performance -------------------------------------------
read -r -p "Set max power mode + jetson_clocks for lowest latency? [y/N] " ans || true
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  sudo nvpmodel -m 0 2>/dev/null || echo "  (nvpmodel not available)"
  sudo jetson_clocks 2>/dev/null || echo "  (jetson_clocks not available)"
fi

# ---- 3. System packages -----------------------------------------------------
echo "== apt deps (qt, python venv, build tools) =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-dev python3-pip build-essential cmake \
  libqt5gui5 libgl1 libglib2.0-0 \
  libusb-1.0-0-dev udev i2c-tools || true

# ---- 3b. Python 3.11 (required by pennylane 0.44; JetPack 5 ships 3.8) -------
if command -v python3.11 >/dev/null 2>&1; then
  PYBIN=python3.11
elif python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  PYBIN=python3
else
  echo "== installing Python 3.11 from the deadsnakes PPA (JetPack 5 ships 3.8) =="
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
  PYBIN=python3.11
fi
echo "Using $PYBIN -> $($PYBIN --version)"

# ---- 3c. sqlite >= 3.35 (chromadb requirement; Ubuntu 20.04 ships 3.31) ------
SQLITE_OK=$($PYBIN - <<'PY'
import sqlite3
maj, mid, _ = (sqlite3.sqlite_version_info + (0, 0, 0))[:3]
print("yes" if (maj, mid) >= (3, 35) else "no")
PY
)
if [ "$SQLITE_OK" != "yes" ]; then
  echo "== building sqlite >= 3.35 into /usr/local (chromadb needs it) =="
  SQLITE_TARBALL=sqlite-autoconf-3460100
  ( cd /tmp \
    && wget -q "https://www.sqlite.org/2024/${SQLITE_TARBALL}.tar.gz" \
    && tar xzf "${SQLITE_TARBALL}.tar.gz" \
    && cd "$SQLITE_TARBALL" \
    && ./configure --prefix=/usr/local >/dev/null \
    && make -j"$(nproc)" >/dev/null \
    && sudo make install >/dev/null \
    && sudo ldconfig )
  echo "   sqlite now: $($PYBIN -c 'import sqlite3; print(sqlite3.sqlite_version)')"
fi

# DepthAI needs a udev rule so the OAK-D Lite is reachable without root
echo "== DepthAI udev rule for OAK-D =="
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true

# ---- 4. Python venv ----------------------------------------------------------
# NOTE: no --system-site-packages — apt's cv2/PyQt5 are built for Python 3.8 and
# would not import in a 3.11 venv anyway; everything comes from pip wheels.
echo "== python venv (.venv, $PYBIN) =="
"$PYBIN" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- 5. App deps (aarch64-friendly) -----------------------------------------
echo "== pip deps =="
python -m pip install -r "$ROOT/requirements-jetson.txt"

# ---- 6. PyTorch --------------------------------------------------------------
# requirements-jetson.txt installs the CPU aarch64 torch wheel from PyPI, which is
# all this app needs (quantum sim = lightning.qubit on CPU; torch = classical layers
# only). NVIDIA's JetPack CUDA wheels are Python-3.8-only and NOT compatible with
# the 3.11 venv — do not mix them in.
if python -c "import torch" 2>/dev/null; then
  echo "== torch present: $(python -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())') =="
else
  echo "!! torch missing — re-run: pip install -r requirements-jetson.txt"
fi

# ---- 7. Verify --------------------------------------------------------------
echo "== verify imports =="
python - <<'PY'
mods = ["depthai","chromadb","pennylane","pennylane_lightning","Jetson.GPIO",
        "numpy","sklearn","cv2","prometheus_client","yaml"]
import importlib
for m in mods:
    try:
        importlib.import_module(m if m != "Jetson.GPIO" else "Jetson.GPIO")
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  MISS {m}  ({type(e).__name__})")
PY

echo ""
echo "== done =="
echo "Activate with:  source $ROOT/.venv/bin/activate"
echo "If torch showed MISS, run the pip command printed above, then re-run this script."
