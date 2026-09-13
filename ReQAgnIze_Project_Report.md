# ReQAgnIze — Quantum Face-Access System
## Detailed Project Report

**Report date:** 2026-09-13
**Prepared from:** laptop `vyomans-shuttle` (Ubuntu 24.04.4) + Jetson Xavier NX (SSH)
**Project package:** `Testing_App_deployment.zip` — sha256 `1ed7c840e4ae8c9c441df2299a0f618d0d8d5bcc7a63f7c7375d497144dfda11` (330,253 bytes, 147 files)
**Deployed copy:** Jetson `~/FR_project/Testing_App_deployment_v2/`
**Companion report:** `JETSON_DEPLOYMENT_REPORT.md` (on both machines)

---

## 1. Executive Summary

ReQAgnIze is an **edge face-access appliance**: a camera feed is checked for liveness (real 3D face, not a photo/screen), passed through two quantum machine-learning circuits, matched against a local vector database, and — on a match — the person's identity is decrypted from post-quantum-encrypted storage and access is granted. No images are stored, and nothing leaves the device.

The `Testing_App_deployment` package contains the complete runtime (application + trained model weights + PQC crypto), self-contained with no external model dependencies.

**Current status:** fully deployed and running on the NVIDIA Jetson Xavier NX with the Luxonis OAK-D Lite camera. A fresh data state was reset on 2026-09-13 (0 registered people); the kiosk is running and ready for new registrations.

---

## 2. What the System Does (User-Facing Flow)

```
IDLE  ──(power on: 'p' key / START / GPIO button)──►  LOADING
                                                         │ (engine + camera warm up once, kept warm)
                                                         ▼
                                            ACTIVE  (live camera preview)
                                                         │ face centered + close
                                                         ▼
                                     LIVENESS (depth 3D + motion, blink/yaw fallback)
                                                         │ passed
                                                         ▼
                                      FACE GATE (10-qubit classifier: face / not-face)
                                                         │ face
                                                         ▼
                              RECOGNIZE (512-dim quantum embedding → ChromaDB + voting)
                                                         │ match ≥ threshold
                                                         ▼
                             DECRYPT identity (PQC-NTRU / AES-256-GCM) → ACCESS ALLOWED
                                                         │
                                                         ▼
                                     RESULT screen (2.5 s) → back to ACTIVE
```

Access events (allowed/denied, score, liveness mode, latency) are appended to a local SQLite log. No face image is ever written to disk.

---

## 3. Architecture

### 3.1 Runtime Pipeline (component map)

```
 OAK-D Lite camera (USB)              plain webcam (fallback)
 RGB + depth(mm), 640×400                index 0
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
             app/camera/{oak,webcam}.py   (backend: auto | oak | webcam)
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
 app/liveness/checks.py         app/recognition/engine.py
 (depth 3D relief + motion;     (Haar detect → 10-qubit face gate →
  blink + head-yaw fallback)     V2 quantum embedder → ChromaDB match)
        │                              │
        └──────────────┬───────────────┘
                       ▼
             app/main.py  (state machine, power, result handling)
                       │
        ┌──────────────┼────────────────┬───────────────┐
        ▼              ▼                ▼               ▼
  app/ui/kiosk.py  app/events/     app/metrics/    app/gpio/io.py
  (cv2 UI)         store.py        exporter.py     (button/LED/relay,
                   (events.db)     (Prometheus)     Jetson.GPIO, guarded)
```

### 3.2 Module Inventory

| Path | Role |
|------|------|
| `app/main.py` | Appliance state machine: OFF → LOADING → ACTIVE → RESULT; warms engine/camera once, runs recognition only after liveness passes; in-app enrollment |
| `app/config.py` | Loads `config.yaml`; path helpers (`abspath`, `qfr_path`, `QFR_PATH` env override) |
| `app/camera/__init__.py` | Backend selection: `auto` → OAK if `depthai` importable, else webcam |
| `app/camera/oak.py` | DepthAI pipeline: 1080p RGB (preview 640×400) + stereo depth aligned to RGB (mm, uint16) |
| `app/camera/webcam.py` | Plain webcam fallback (laptop testing) |
| `app/liveness/checks.py` | Liveness state machine: mode `fast/balanced/strict`; depth variance in the face box (primary) + frame motion; blink (EAR) and head-yaw fallback |
| `app/liveness/landmarks.py` | Face landmarks provider (mediapipe if available, guarded) |
| `app/recognition/engine.py` | Heavy engine: loads models once; `enroll()` and `recognize()`; duplicate-registration guard; NTRU metadata storage |
| `app/ui/kiosk.py` | OpenCV kiosk UI: idle / loading / active / result screens; `_force_opaque()` fix for Xwayland compositing |
| `app/events/store.py` | SQLite access log (`events.db`) — user, allowed, score, votes, liveness, latency; no images |
| `app/metrics/exporter.py` | Optional Prometheus exporter (port 9108) |
| `app/gpio/io.py` | Optional physical power button / status LED / door relay via `Jetson.GPIO` (guarded, mock mode otherwise) |
| `vendor/src/` | V1 building blocks: Haar face detection, preprocessing, 10-qubit QCNN architecture, quantum encoding (PCA + RY/RZ) |
| `vendor/src_v2/` | V2 recognizer: 3-level quantum Haar-wavelet features, 8-qubit hierarchical QCNN, 512-dim embedder, ChromaDB wrapper with voting |
| `vendor/src_v3/` | 10-qubit face/non-face gate wrapper (`classify_face`) |
| `vendor/PQC/` | Post-quantum crypto: NTRU (keygen/encrypt/decrypt/ring math) + AES-256-GCM metadata storage |
| `vendor/models/` | 4 trained artifacts (see 3.3) |
| `scripts/` | `selfcheck.py`, `enroll.py`, `test_camera.py`, `run.sh`/`run_jetson.sh`, `setup_jetson.sh`, `identify_board.sh` |
| `tests/` | Pytest logic tests: liveness state machine + event store |
| `config.yaml` | All tunables (camera, model, liveness, GPIO, events, metrics) |

### 3.3 Quantum Components

| Component | Qubits | Params | Input → Output | Weights |
|-----------|--------|--------|----------------|---------|
| **V2 recognizer embedder** (Quantum Haar-Wavelet) | 8 | 34 | face ROI → 64 windows × 3-level Haar (LL/LH/HL/HH) → 512-dim embedding | `vendor/models/qcnn_v2_trained.pth` (14.8 KB) |
| **V1 face/non-face gate** | 10 | 25 | 16×16×3 ROI → PCA 768→10 → RY/RZ encoding → softmax(face/non-face) | `vendor/models/qcnn_classifier.pth` (11.1 KB) + `pca_model.pkl` + `scaler.pkl` |
| **Matching** | — | — | ChromaDB (cosine / HNSW), top-5 similarity voting, threshold 0.75, min_votes 2 | `vector_db/` |
| **PQC metadata** | — | — | NTRU (wrap) + AES-256-GCM (name/phone/age) | `vendor/PQC/keys/*.pkl` |

Quantum backend: **`lightning.qubit` 0.44.0** (C++ state-vector) with `pennylane` 0.44.0 — verified working on the Jetson under the libgomp preload fix (section 6.2).

### 3.4 Security & Privacy Properties

- On-device only: no cloud calls required (`events.sync_enabled: false` by default).
- Images are never persisted: frames live in RAM; enrollment shots are converted to embeddings and discarded.
- Identity data at rest: 512-dim embedding templates (ChromaDB) + PQC-encrypted PII (`*.meta.enc`).
- Post-quantum: NTRU key encapsulation with AES-256-GCM for the payload.
- Liveness anti-spoof: OAK-D stereo depth (3D relief + motion) as primary; blink + head-yaw as fallback.
- Note: `vendor/PQC/keys/private_key.pkl` ships/permissions as `rw-rw-r--`; restrict to `600` on production devices.

---

## 4. Where Data Is Stored (Jetson, relative to `~/FR_project/Testing_App_deployment_v2/`)

| Data | Path | Format / Notes |
|------|------|----------------|
| Face embeddings (templates) | `vector_db/` | ChromaDB: `chroma.sqlite3` + HNSW segment `.bin` files; collection `face_embeddings_v2` |
| Pickle backup of embeddings | `models/recognition_db_v2.pkl` | Written on every enrollment (`save_database()`); loaded if present |
| Encrypted PII (name/phone/age) | `vendor/PQC/encrypted_metadata/<user>.meta.enc` | NTRU + AES-256-GCM |
| NTRU keypair | `vendor/PQC/keys/{public,private}_key.pkl` | Generated on first run if missing |
| Access audit log | `events.db` | SQLite: ts, device, user, allowed, score, votes, liveness, latency — no images |
| Performance log | `vendor/logs/timing_log.csv` | Per-stage timings |

**Reset all enrollment data** (keys can stay): delete `vector_db/`, `models/recognition_db_v2.pkl`, `events.db`, `vendor/PQC/encrypted_metadata/*.meta.enc`, `vendor/logs/timing_log.csv`.
**Never delete keys and encrypted metadata separately** unless doing a full reset — old `*.meta.enc` files would become undecryptable.

---

## 5. Hardware & Environment

| Item | Details |
|------|---------|
| Target board | NVIDIA Jetson Xavier NX Developer Kit (p3449-0000 + p3668-0001) |
| OS / BSP | JetPack 5.1.1 — L4T R35.3.1, kernel 5.10.104-tegra, aarch64 |
| Camera | Luxonis OAK-D Lite (Intel Movidius MyriadX), USB ID `03e7:2485`, MX ID `19443010B15A7A2700`, currently Bus 001 (USB 2.0); works, prefer USB3 for max fps |
| Connection | USB device-mode network: Jetson `192.168.55.1`, laptop `192.168.55.100`; access via `ssh jetson` (user `auvmu`) |
| Display | GNOME on Xwayland `:0` (kiosk renders on the Jetson's screen) |
| Python | 3.11.0 (`/usr/local/bin/python3.11`) |
| Venv | `~/FR_project/.venv` (2.9 GB) shared by symlink into v2 |
| Key packages | torch 2.2.2 (CPU), pennylane 0.44.0, pennylane-lightning 0.44.0, chromadb 1.4.0, depthai 2.24.0.0, opencv-contrib-python 4.11.0.86, scikit-learn 1.7.2, joblib 1.5.3, cryptography 46.0.3, PyYAML 6.0.3, prometheus_client 0.21.1, Jetson.GPIO 2.1.13, numpy 1.26.4 |
| Dev machine | Ubuntu 24.04.4 laptop (control/analysis; package also verified byte-identical here) |

---

## 6. Deployment History — What Was Done

### 6.1 Package analysis and target selection
- Identified the package as Linux-capable by design (README Jetson section, `scripts/setup_jetson.sh`, `identify_board.sh`); the OAK-D is the intended camera.
- Located the actual camera on the **Jetson**, not the laptop (`lsusb` `03e7:2485`), and used the USB gadget network to work over SSH.
- Both zip copies (laptop and Jetson) hash-identical — no transfer needed.

### 6.2 Problems found and fixed

| # | Problem | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | DepthAI: `Insufficient permissions to communicate with X_LINK_UNBOOTED device` | `/etc/udev/rules.d/80-movidius.rules` was malformed (`SUBSYSTEM==usb, ATTRS{idVendor}==03e7, MODE=0666` — no quotes, single `=`), so udev ignored it; USB node stayed root-only | Rewrote rule: `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"` + `udevadm control --reload-rules && udevadm trigger` (one-time, survives reboot) |
| 2 | Kiosk load failed: `Pre-compiled binaries for lightning.qubit are not available` — while `selfcheck` passed | aarch64 static TLS: the kiosk's Qt/cv2 window loads before `lightning_qubit_ops`, whose bundled `libgomp-947d5fa1.so` then cannot dlopen (`cannot allocate memory in static TLS block`); PennyLane surfaces it as the misleading "pre-compiled binaries" error | Added `_preload_lightning_libgomp()` to `app/__init__.py` — dlopens the bundled libgomp before any window exists (covers kiosk, enroll, all entry points) |
| 3 | Kiosk window transparent/incomplete over SSH | Missing `DISPLAY` + Qt platform under Xwayland; compositor alpha | Ported proven `phase_1` fixes: `_force_opaque()` in `app/ui/kiosk.py`, `run_jetson.sh`, `scripts/run.sh` (`DISPLAY=:0`, `QT_QPA_PLATFORM=xcb`, `_NET_WM_BYPASS_COMPOSITOR`) |

### 6.3 Setup steps performed
1. Fresh extraction of the zip to `~/FR_project/Testing_App_deployment_v2` (inner folder stripped; stale `__pycache__` removed).
2. `.venv` symlinked to the existing provisioned `~/FR_project/.venv` (no reinstall; the zip's `requirements-app.txt` pins are x86/laptop builds and must not be pip-installed on ARM64).
3. Config switched to device mode: `camera.backend: "auto"`, `liveness.local_test_mode: false`.
4. Display fixes ported from `phase_1` (verified working there).
5. `app/__init__.py` libgomp preload fix added.
6. udev rule corrected.

### 6.4 Verification evidence

| Check | Result |
|-------|--------|
| `scripts/selfcheck.py --no-camera` | `RESULT: PASS` — all libraries, vendored weights, engine, both quantum circuits |
| `pytest tests -q` | `3 passed` |
| OAK-D headless capture test | 15 RGB frames (640×400×3) + 14 depth frames (uint16 mm, median 1545 mm) — `OAK TEST: PASS` |
| Live kiosk | OAK feed live, face detected, real depth liveness passed, recognition pipeline ran |
| Access log during testing (pre-reset) | `user=Chanti allowed=1 score=0.943/0.934/0.932/0.923 liveness=fast` |
| Error log | none |

### 6.5 Data reset (fresh start)
- Backup taken: `~/FR_project/backups/reqagnize_v2_data_20260913_100150.tar.gz` (28 KB — contains the 3 old embeddings, pickle backup, 17 events, encrypted metadata, logs).
- Wiped: `vector_db/`, `models/recognition_db_v2.pkl`, `events.db`, `vendor/PQC/encrypted_metadata/*.meta.enc`, `vendor/logs/timing_log.csv`.
- Kept: NTRU keys.
- Verified after reset: selfcheck `RESULT: PASS` with `ChromaDB collection (0 enrolled embeddings)`; kiosk restarted fresh.

---

## 7. Current State (2026-09-13)

- Kiosk running on the Jetson via `./run_jetson.sh` (fresh database, idle screen).
- 0 registered people — ready for new enrollments.
- Engine uses `lightning.qubit`; OAK-D RGB + depth active; real depth liveness (`local_test_mode: false`).
- Working folder: `~/FR_project/Testing_App_deployment_v2/` (Jetson). The earlier `phase_1/` and the Sep 1 extraction still contain their old data and are untouched.

---

## 8. How to Operate

### Run the kiosk
```bash
cd ~/FR_project/Testing_App_deployment_v2
./run_jetson.sh
```
Power on with `p` (window focused) or the START button. Quit with `q`.

### Register a person
- In kiosk: power on → **REGISTER** → type name / phone / age → SPACE ×3 to capture.
- Or CLI (stop the kiosk first — the OAK-D is exclusive):
```bash
pkill -f "python -m app.main"
source ~/FR_project/.venv/bin/activate
cd ~/FR_project/Testing_App_deployment_v2
python scripts/enroll.py --name "Name" --phone 9999999999 --age 25
```
Duplicate faces are rejected (`Already registered as '<name>'`).

### Verify components
```bash
cd ~/FR_project/Testing_App_deployment_v2
.venv/bin/python scripts/selfcheck.py --no-camera
.venv/bin/python -m pytest tests -q
```

### Reset data / backup
```bash
# wipe enrollments (keep keys)
rm -rf vector_db events.db models/recognition_db_v2.pkl
rm -f vendor/PQC/encrypted_metadata/*.meta.enc vendor/logs/timing_log.csv
# backup
tar -czf ~/FR_project/backups/data_$(date +%F_%H%M%S).tar.gz vector_db models/recognition_db_v2.pkl events.db vendor/PQC vendor/logs
```

---

## 9. Known Issues, Caveats, and Recommendations

1. **`vendor/src_v2` hard-codes `lightning.qubit`** with no runtime fallback (`qcnn_recognition_v2.py:40`, `quantum_haar_wavelet.py:59`). It works on this Jetson only with the libgomp preload fix. A fallback-enabled variant (`_get_quantum_device()`) exists in the laptop's `~/Quantum_project/Quantum/vendor/` — port it if the preload fix is ever removed, or for other boards.
2. **`scripts/setup_jetson.sh` in the zip references `requirements-jetson.txt`, which is not inside the zip** (exists only in `~/Quantum_project/Quantum/`). Do not run it blindly; provide that file first or reuse the existing venv.
3. **Do not pip-install `requirements-app.txt` on ARM64** — the pins are x86 wheels (torch 2.2.2, opencv-contrib, pennylane-lightning). Use the JetPack/Jetson torch wheel and the existing venv.
4. **OAK-D currently on USB 2.0** — functional at reduced fps; move to a USB 3 port for best throughput.
5. **Private NTRU key permissions** are `rw-rw-r--` — set `chmod 600 vendor/PQC/keys/private_key.pkl` for production.
6. **`config.yaml` flags with no implementation in this package:** `camera.on_vpu_detection` (detection runs host-side) and `events.store_thumbnail` (no thumbnail code — images never stored). `gpio.enabled` and `metrics.enabled` are functional (Jetson.GPIO present).
7. **Key + metadata pairing:** if you ever delete keys without deleting `encrypted_metadata/`, existing identities become undecryptable; delete them together.

---

## 10. Deliverables & Artifact Locations

| Artifact | Location |
|----------|----------|
| Deployed app (working) | Jetson: `~/FR_project/Testing_App_deployment_v2/` |
| Deployment report | Jetson: `~/FR_project/Testing_App_deployment_v2/JETSON_DEPLOYMENT_REPORT.md`; laptop: `~/Downloads/JETSON_DEPLOYMENT_REPORT.md` |
| This report | laptop: `~/Downloads/ReQAgnIze_Project_Report.md` |
| Source package (zip) | laptop: `~/Downloads/Testing_App_deployment.zip`; Jetson: `~/FR_project/Testing_App_deployment.zip` |
| Data backup (pre-reset) | Jetson: `~/FR_project/backups/reqagnize_v2_data_20260913_100150.tar.gz` |
| Original full project (with training code/tools) | laptop: `~/Quantum_project/Quantum/` |

---

## Appendix A — Configuration Highlights (`config.yaml`, deployed values)

```yaml
paths:    qfr_path: "vendor"
device:   id: "reqagnize-edge-01"; display: 800×480, windowed
camera:   backend: "auto"; rgb_fps: 20; rgb_resolution: "1080p"; depth_enabled: true
model:    version: "v2"; quantum_device: "lightning.qubit"; recognition_threshold: 0.75;
          min_votes: 2; classify_enabled: true; classify_threshold: 0.50; embedding_dim: 512
liveness: mode: "fast"; local_test_mode: false; depth_primary: true;
          depth_strong_variance_mm: 18.0; depth_min_variance_mm: 8.0; timeout_sec: 12.0
gpio:     enabled: false (no physical button/LED/relay wired in this setup)
events:   sqlite_path: "events.db"; sync_enabled: false; store_thumbnail: false
metrics:  enabled: false; port: 9108
```

## Appendix B — Exact Error Text Captured (for future debugging)

```
# 1) OAK permission
[depthai] [warning] Insufficient permissions to communicate with X_LINK_UNBOOTED device having name "1.2.1". Make sure udev rules are set

# 2) Kiosk load failure (real cause)
UserWarning: .../pennylane_lightning.libs/libgomp-947d5fa1.so.1.0.0: cannot allocate memory in static TLS block
ImportError: Pre-compiled binaries for lightning.qubit are not available. To manually compile from source, follow the instructions at https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html.
```

## Appendix C — Key Commands Reference

```bash
# SSH to the Jetson over USB
ssh jetson                      # auvmu@192.168.55.1

# OAK-D sanity check
~/FR_project/.venv/bin/python -c "import depthai as dai; print([(d.getMxId(), str(d.state)) for d in dai.Device.getAllAvailableDevices()])"

# udev rule (only needed once, requires sudo)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
