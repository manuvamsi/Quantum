# Benchmark Report — Testing_App_deployment_v3 (ReQAgnIze Edge Appliance)

**Board:** NVIDIA Jetson Xavier NX Dev Kit (6× Cortex-A78 @ 1420 MHz, 7 GB RAM)
**OS:** L4T R35.3.1 (JetPack 5.3.1), Python 3.11, aarch64
**Stack:** torch 2.2.2 (CPU) · PennyLane 0.44.0 (lightning.qubit) · OpenCV 4.11 · ChromaDB 1.4.0 · depthai 2.24.0.0
**App commit:** `1e94723` (github.com/roshan5619/Testing_App_Deployment)
**Method:** external harness `benchmark_v3/bench.py` (repo code untouched) + `tegrastats` @ 1 kHz
**Date:** 2026-09-17

---

## 1. Run location: 100% CPU

| Component | Device | Evidence |
|---|---|---|
| PyTorch QCNN layers | CPU | CPU-only wheel; `torch.cuda.is_available() = False`; weights loaded `map_location='cpu'` |
| Quantum circuits (8-qubit embedder, 10-qubit gate) | CPU | `lightning.qubit` (C++ CPU simulator); `lightning.gpu` not installed (DeviceError) |
| Face detection | CPU | OpenCV Haar cascade, single-threaded |
| Liveness landmarks | CPU | mediapipe with XNNPACK (CPU) delegate |
| NVIDIA GPU (GR3D) | idle | `GR3D_FREQ 0%` across the entire bench |

Why no GPU: CPU-only torch build installed (JetPack CUDA wheel never installed); PennyLane GPU backend
(CuQuantum) absent; code hardcodes CPU devices; and the quantum circuits are tiny (8–10 qubits =
≤1 KB state), so GPU transfer overhead would outweigh the compute.

## 2. Load / boot timing

| Stage | Time |
|---|---|
| Library imports: numpy 0.39 · cv2 0.15 · torch 4.25 · pennylane 5.83 · chromadb 2.60 · sklearn 1.48 | ~14.7 s |
| Vendor model imports (QCNN + Haar-wavelet circuits) | 13.3 s |
| FaceRecognizer init (ensemble weights + PCA/scaler + ChromaDB) | 2.6 s |
| FaceDetector (Haar XML) / FacePreprocessor | 0.09 s / ~0 s |
| Circuit warmup (2 passes) | 0.5 s |
| **Cold power-on → READY (NTRU keys present)** | **~17 s** |
| First-ever run: NTRU key generation (one-time) | +90.7 s |
| `engine.load()` when libs already imported | 1.9 s |

## 3. Per-recognition latency (detect → face gate → embed → match)

| Step | @480p | @1080p |
|---|---|---|
| Haar face detection | 128 ms (p95 133) | **800 ms** ← dominant |
| 10-qubit face/non-face gate | 32 ms (p95 45) | 32 ms |
| Ensemble 512-dim quantum embedding | 90 ms (p95 98) | 90 ms |
| ChromaDB top-5 cosine query | 8 ms | 8 ms |
| **Total per attempt** | **~250 ms** | ~920 ms |

First-call overheads (circuit compile): embed 0.41 s, classifier 0.03 s — absorbed by warmup at boot.

## 4. Enrollment & crypto

| Step | Time |
|---|---|
| NTRU encrypt (per 512-d embedding, N=509/p=3/q=2048) | 434 ms |
| NTRU decrypt (per unlock) | 0.2 ms |
| ChromaDB upsert ×100 embeddings | 150 ms |
| Full 3-shot enroll (detect + gate + embed + encrypt ×3) | ~2 s |

## 5. Resource utilization (tegrastats)

- Inference pegs **1 core at ~98%** @1420 MHz; other 5 cores idle (quantum sim + Haar are single-threaded)
- GPU 0%, EMC ~0%, RAM 2.6–2.9 GB / 6.8 GB used, temps ~43–44 °C, no throttling

## 6. Functional verification

- `scripts/selfcheck.py --no-camera` → **RESULT: PASS** (engine, PQC keys, both quantum circuits)
- `pytest tests -q` → **3 passed**
- Live kiosk on OAK-D Lite: running (requires `camera.backend: "auto"` on Jetson)

## 7. Takeaways for small-board migration

- Detection dominates the frame loop → run kiosk at 480p (~4 FPS attempts) or offload to the
  OAK-D VPU (`on_vpu_detection: true` — already configured)
- The quantum pipeline is mathematically tiny (256–1024 amplitudes) — portable to pure C, fits
  sub-$50 SBCs / even FPGA fabric; no GPU dependency anywhere in the current stack
- Python stack footprint (~3 GB venv, ~2.7 GB RAM) is the only thing pinning this to big boards;
  a C/C++ port of gate+embed+match+NTRU would run in <128 MB RAM at 2–4 W
