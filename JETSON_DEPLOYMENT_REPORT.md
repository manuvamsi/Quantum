# ReQAgnIze — Jetson Deployment Report

**Date:** 2026-09-13
**Folder:** `~/FR_project/Testing_App_deployment_v2` (Jetson)
**Source:** `~/FR_project/Testing_App_deployment.zip`
**sha256:** `1ed7c840e4ae8c9c441df2299a0f618d0d8d5bcc7a63f7c7375d497144dfda11` (byte-identical copy exists on the laptop)

---

## 1. Hardware / Software

| Item | Value |
|------|-------|
| Board | NVIDIA Jetson Xavier NX Developer Kit |
| JetPack / L4T | 5.1.1 / R35.3.1 (kernel 5.10, aarch64) |
| Camera | Luxonis OAK-D Lite — USB `03e7:2485` (Intel Movidius MyriadX), MXID `19443010B15A7A2700` |
| Access | `ssh jetson` → `auvmu@192.168.55.1` (USB device-mode network) |
| Python | 3.11.0 (`/usr/local/bin/python3.11`) |
| Environment | `~/FR_project/.venv` (pre-provisioned): torch 2.2.2 (CPU), pennylane 0.44 + pennylane-lightning 0.44, chromadb 1.4.0, depthai 2.24.0.0, opencv-contrib 4.11, scikit-learn, Jetson.GPIO 2.1.13 |
| Display | GNOME session, Xwayland `:0` (kiosk windows render on the Jetson screen) |

The deployed app: camera → Haar detect → liveness (OAK depth primary) → 10-qubit face gate → V2 Quantum Haar-Wavelet recognition (512-dim, ChromaDB) → PQC-NTRU encrypted identity → ACCESS ALLOWED / DENIED.

---

## 2. What Was Created

Fresh extraction of the zip into a new folder (no modification of `phase_1/` or the Sep 1 extraction):

- `~/FR_project/Testing_App_deployment_v2/` — app root (zip's inner folder stripped)
- `.venv` → symlink to `~/FR_project/.venv` (reuses the 2.9 GB provisioned venv; no reinstall)
- Shipped stale `__pycache__` removed

Config changes in `config.yaml`:

| Key | Before | After |
|-----|--------|-------|
| `camera.backend` | `"webcam"` | `"auto"` (OAK-D if present) |
| `liveness.local_test_mode` | `true` | `false` (real depth + gesture anti-spoof) |

---

## 3. Problems Found and Fixed

### 3.1 Malformed udev rule → OAK-D permission denied

**Symptom (DepthAI):**
```
Insufficient permissions to communicate with X_LINK_UNBOOTED device having name "1.2.1"
```
**Root cause:** `/etc/udev/rules.d/80-movidius.rules` contained an invalid rule (single `=`, unquoted values), so udev silently ignored it and the USB node stayed `root:root 664`:
```
SUBSYSTEM==usb, ATTRS{idVendor}==03e7, MODE=0666     # broken
```
**Fix (needs sudo):**
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
**Result:** `/dev/bus/usb/001/008` → `666`, DepthAI lists the device, pipeline opens.

### 3.2 Kiosk load failure: "Pre-compiled binaries for lightning.qubit are not available"

**Symptom:** `./run_jetson.sh` showed `Load failed: Pre-compiled binaries for ligh…` on the loading screen.

**Actual cause (aarch64 static TLS bug, not a missing wheel):**
```
UserWarning: …/pennylane_lightning.libs/libgomp-947d5fa1.so.1.0.0:
             cannot allocate memory in static TLS block
ImportError: Pre-compiled binaries for lightning.qubit are not available.
```
`lightning_qubit_ops.so` needs its bundled libgomp, which uses initial-exec TLS. The kiosk creates the Qt/cv2 window first; by the time the engine loads, the static TLS block is full and dlopen fails. Standalone checks (no window) passed, which is why `selfcheck.py` worked.

**Fix — `app/__init__.py` (runs before any window for every entry point):**
```python
def _preload_lightning_libgomp() -> None:
    if not (sys.platform == "linux" and platform.machine() == "aarch64"):
        return
    import importlib.util
    spec = importlib.util.find_spec("pennylane_lightning")
    if not spec or not spec.submodule_search_locations:
        return
    pkg_dir = os.path.dirname(list(spec.submodule_search_locations)[0])
    for lib in sorted(glob.glob(os.path.join(pkg_dir, "pennylane_lightning.libs", "libgomp-*.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

_preload_lightning_libgomp()
```
**Verified:** same window-first repro → `ENGINE LOAD OK`; live kiosk runs with `lightning.qubit` (fast C++ backend).

### 3.3 Display rendering under Xwayland/SSH (carried over from phase 1)

Ported the proven phase-1 fixes so the new folder renders correctly:
- `app/ui/kiosk.py` — `_force_opaque()` (`_NET_WM_BYPASS_COMPOSITOR` via xdotool/xprop, only when running from SSH/tty)
- `run_jetson.sh` — sets `DISPLAY=:0` + `QT_QPA_PLATFORM=xcb`, activates `.venv`
- `scripts/run.sh` — same DISPLAY/QT env handling

---

## 4. Verification Evidence

| Check | Result |
|-------|--------|
| `scripts/selfcheck.py --no-camera` | `RESULT: PASS` (engine, weights, ChromaDB 3 enrollments, both quantum circuits) |
| `pytest tests -q` | `3 passed` |
| OAK-D headless frame test | 15 RGB frames (640×400×3) + 14 depth frames (uint16 mm, median 1545 mm) — `OAK TEST: PASS` |
| Live kiosk (`./run_jetson.sh`, press `p`/START) | OAK feed live, face detected, real depth liveness passed |
| `events.db` (recognitions while testing) | `user=Chanti allowed=1 score=0.943/0.934/0.932/0.923 liveness=fast` |
| App log | no errors/tracebacks |

---

## 5. How to Run It

```bash
cd ~/FR_project/Testing_App_deployment_v2
./run_jetson.sh                 # kiosk on the Jetson display (:0)
```
- Power on: press `p` on the kiosk window or click **START**
- Enroll: **REGISTER** in the kiosk, or
  ```bash
  source ~/FR_project/.venv/bin/activate
  python scripts/enroll.py --name "Name" --phone 9999999999 --age 25
  ```
- Quit kiosk: `q`
- Behind the scenes: `python -m app.main` inside the venv (with `DISPLAY=:0`, `QT_QPA_PLATFORM=xcb`)

Stop / restart:
```bash
pkill -f "python -m app.main"
cd ~/FR_project/Testing_App_deployment_v2 && nohup ./run_jetson.sh > /tmp/kiosk.log 2>&1 &
```

---

## 6. Redeploy Checklist (fresh machine / fresh extraction from zip)

1. Extract the zip (fresh copy; do not modify `phase_1/`).
2. Reuse or provision the venv. Do **not** `pip install -r requirements-app.txt` on Jetson — those pins are x86/laptop builds. On ARM64: torch from the JetPack/Jetson wheel index, `cv2` from apt or the existing venv, plus `depthai`, `Jetson.GPIO`, `mediapipe`.
3. **Critical:** add the `_preload_lightning_libgomp()` fix to `app/__init__.py` (section 3.2) — without it, the kiosk fails at load while standalone scripts pass.
4. Port the display fixes (`kiosk.py`, `run_jetson.sh`, `scripts/run.sh`).
5. `config.yaml`: `camera.backend: "auto"`, `liveness.local_test_mode: false`.
6. udev rule (one-time, survives reboots): section 3.1.
7. Verify in order: `selfcheck.py --no-camera` → `oak` frame test → `./run_jetson.sh`.

## 7. Gotchas

- `scripts/setup_jetson.sh` in the zip references `requirements-jetson.txt`, which is **not inside the zip** (it exists only in `~/Quantum_project/Quantum` on the laptop). Don't run it blindly on the Jetson until that file is provided.
- `vendor/src_v2/qcnn_recognition_v2.py` hard-codes `qml.device("lightning.qubit")` with no fallback. It works on this Jetson **only because of the preload fix**. The version with a runtime fallback (`_get_quantum_device()`) exists in the laptop's `~/Quantum_project/Quantum/vendor/`. If the preload fix is ever removed, port that fallback.
- The `camera.on_vpu_detection: true` flag is present in config but the current `oak.py` runs detection host-side; no action needed for the working setup.
- OAK-D is on Bus 001 (USB 2.0 on this board); it works, but for maximum fps/depth quality prefer a USB 3 port if free.
- sudo password is required for udev installs; it has been done and persists across reboots.

---

*Working state as of 2026-09-13: kiosk PID running via `./run_jetson.sh`; engine `lightning.qubit`, OAK-D RGB+depth, depth liveness, PQC recognition all verified live.*


---

## 8. Fix Log — 2026-09-16: Recognition Discrimination Fix (hybrid embedding)

**Problem:** every face (registered or not) was recognized as the single enrolled user, and new
registrations were blocked with "Already registered as '<name>'".

**Root cause:** the pure quantum embedding (Haar-wavelet stats → 8-qubit QCNN, 512-dim) is nearly
input-independent — random noise scored ~0.95 cosine against an enrolled face; impostor faces scored
0.90–0.96, overlapping the genuine band. No threshold could separate people.

**Fix:** hybrid embedding (`model.embedding_backend: "hybrid"`) — 1024-dim signature =
0.85 × InceptionResNetV1 (VGGFace2) + 0.15 × quantum QCNN (both L2-normalized).
Threshold recalibrated 0.75 → 0.50 (impostor max ≈ 0.39, genuine min ≈ 0.60, calibrated 2026-09-16).
New collection `face_embeddings_v3_hybrid`; the old 512-dim DB was wiped — re-enrollment required.
Fallback to quantum-only if `facenet_pytorch` is missing.

**Verification:** 8/8 impostors denied (0.0–0.35); registered person accepted across variants
(0.71–0.99); new registration works; duplicate check still blocks; selfcheck PASS; pytest 3 passed.
Released as `v2.0.1` on GitHub.
