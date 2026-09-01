#!/usr/bin/env bash
# identify_board.sh — run this ON THE JETSON and paste the full output back.
# It reports the board model, JetPack/L4T version, CUDA, Python, memory, disk,
# and whether the key libraries are already present. This tells us exactly which
# aarch64 PyTorch / PennyLane wheels to use in setup_jetson.sh.

set -u
line() { printf '%s\n' "------------------------------------------------------------"; }

echo "==================  JETSON BOARD REPORT  ==================="

line
echo "[ Board model ]"
if [ -f /proc/device-tree/model ]; then
  tr -d '\0' < /proc/device-tree/model; echo
else
  echo "unknown (no /proc/device-tree/model)"
fi

line
echo "[ L4T / JetPack ]"
if [ -f /etc/nv_tegra_release ]; then
  head -1 /etc/nv_tegra_release
else
  echo "no /etc/nv_tegra_release"
fi
# JetPack meta-package version (maps L4T -> JetPack)
dpkg-query --show nvidia-l4t-core 2>/dev/null || true
apt-cache show nvidia-jetpack 2>/dev/null | grep -m1 Version || echo "nvidia-jetpack meta not found (that's OK)"

line
echo "[ OS / kernel / arch ]"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
echo "kernel: $(uname -r)   arch: $(uname -m)"

line
echo "[ CUDA ]"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -1; else echo "nvcc not on PATH"; fi
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"

line
echo "[ Python ]"
for py in python3 python3.8 python3.9 python3.10; do
  command -v "$py" >/dev/null 2>&1 && echo "$py -> $($py --version 2>&1)"
done
echo "pip3: $(python3 -m pip --version 2>&1)"

line
echo "[ Memory / power ]"
free -h | awk 'NR==1||NR==2'
command -v nvpmodel >/dev/null 2>&1 && sudo nvpmodel -q 2>/dev/null | sed 's/^/nvpmodel: /' || echo "nvpmodel not available"

line
echo "[ Disk / mounts ]"
df -h / /mnt /media 2>/dev/null | awk 'NR==1 || /\/$|\/mnt|\/media/'
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null | grep -Ei 'nvme|sd|mmc' || true

line
echo "[ Existing python libs (already installed?) ]"
python3 - <<'PY' 2>/dev/null || echo "python import probe failed"
import importlib.util as u
for m in ["torch","torchvision","pennylane","pennylane_lightning","cv2","chromadb","depthai","Jetson.GPIO","numpy","sklearn","PyQt5","prometheus_client"]:
    print(f"  {m:22s} {'FOUND' if u.find_spec(m.replace('.', '/').split('/')[0]) else 'missing'}")
PY

line
echo "[ USB devices (look for Luxonis / MyriadX for the OAK-D Lite) ]"
lsusb 2>/dev/null | grep -Ei 'luxonis|movidius|myriad|03e7' || echo "  OAK-D not detected on USB right now (plug it into a USB3 port)"

line
echo "[ GPIO header ]"
ls /sys/class/gpio 2>/dev/null | head || echo "no /sys/class/gpio"
python3 -c "import Jetson.GPIO as G; print('  Jetson.GPIO model:', G.gpio_pin_data.get_data()[0])" 2>/dev/null || echo "  Jetson.GPIO not installed yet"

line
echo "==================  END REPORT  ==================="
echo "Paste everything above back so the exact wheels can be pinned."
