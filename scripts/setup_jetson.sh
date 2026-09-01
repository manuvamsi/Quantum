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
echo "== apt deps (opencv, qt, python venv, build tools) =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-dev python3-pip build-essential cmake \
  python3-opencv python3-pyqt5 libqt5gui5 \
  libusb-1.0-0-dev udev i2c-tools || true

# DepthAI needs a udev rule so the OAK-D Lite is reachable without root
echo "== DepthAI udev rule for OAK-D =="
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true

# ---- 4. Python venv (system-site so we get apt's cv2/PyQt5/CUDA) ------------
echo "== python venv (.venv, --system-site-packages) =="
python3 -m venv --system-site-packages "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- 5. App deps (aarch64-friendly) -----------------------------------------
echo "== pip deps =="
python -m pip install -r "$ROOT/requirements-jetson.txt"

# ---- 6. PyTorch (JetPack-specific) — detect, then guide ---------------------
if python -c "import torch" 2>/dev/null; then
  echo "== torch already present: $(python -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())') =="
else
  echo "!! PyTorch is NOT installed and must match $JP (L4T $L4T)."
  echo "   Recommended: NVIDIA's Jetson wheel index (community mirror jetson-ai-lab):"
  case "$L4T_MAJOR" in
    36) echo "     pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126" ;;
    35) echo "     pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp5/cu114" ;;
    32) echo "     Use NVIDIA's JP4 torch wheel from developer.download.nvidia.com (Python 3.6/3.8)." ;;
    *)  echo "     Unknown L4T — paste identify_board.sh output and I'll pin the exact wheel." ;;
  esac
  echo "   (torch is only needed for the classical QCNN layers; the quantum sim runs on CPU.)"
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
