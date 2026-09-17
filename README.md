
## 1. Requirements

- **Python 3.10 or 3.11** (3.11 recommended — matches the pinned wheels).
- A **webcam** for the live test.


Check your Python:
```bash
python --version        # macOS/Linux may need: python3 --version
```

---

## 2. Setup (create a venv, then install)

**Windows (PowerShell)**
```powershell
cd Testing_App_deployment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-app.txt
```

**macOS / Linux**
```bash
cd Testing_App_deployment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-app.txt
```

> **macOS note:** if a pinned version has no wheel for your machine (e.g. Apple Silicon), loosen
> the pin — change that line in `requirements-app.txt` from `==` to `>=` and reinstall. `torch`,
> `opencv-contrib-python`, and `chromadb` all publish macOS wheels.
>
> **Linux note:** the GUI windows (enroll / kiosk) need a desktop session. Headless server? Run
> only the headless checks in step 3.

---

## 3. Verify it works — headless (no camera, no face images)

Run these from the `Testing_App_deployment/` folder with the venv active.

```bash
python scripts/selfcheck.py          # full component check (add --no-camera on a headless box)
python -m pytest tests -q            # logic tests (liveness state machine + event store)
```

`selfcheck.py` should end with **`RESULT: PASS`** — it confirms every library imports, the
vendored weights are present, the engine loads (**V2 recognizer + 10-qubit classifier + PQC-NTRU
keys + ChromaDB**), and **both quantum circuits actually run** on synthetic data (classifier
returns a probability; the embedder returns a 512-dim vector). The first load takes ~15–40 s
(it warms the circuits and generates fresh NTRU keys under `vendor/PQC/keys/`).

`pytest` should print **`3 passed`**.

> If `selfcheck.py` says *FAIL — install requirements first*, redo step 2 inside the venv.

---

## 4. Verify it works — live (your own face, webcam)

```bash
# enroll yourself (a window opens; press SPACE 3 times to capture, then it saves)
python scripts/enroll.py --name "Your Name" --phone 9999999999 --age 25

# run the kiosk, then press 'p' (or it auto-starts) and look at the camera
python -m app.main
```

Expected: after liveness + the face gate, the screen shows **`ACCESS ALLOWED — Your Name`**.
Enrolling the same face again is blocked with *"Already registered"* (duplicate check).
Press **`q`** to quit the kiosk, **`e`** to open in-app enrollment.

Convenience launchers: `scripts/run.sh` (macOS/Linux) or
`powershell -ExecutionPolicy Bypass -File scripts\run.ps1` (Windows).

Your data stays on this machine: embeddings in `vector_db/`, encrypted PII under
`vendor/PQC/`, access events in `events.db`.

---

## 5. What's in here

```
Testing_App_deployment/
├── app/                     # the runtime application (state machine, camera, liveness,
│                            #   recognition engine, kiosk UI, gpio/metrics/events — all guarded)
├── vendor/                  # SELF-CONTAINED model code + weights (no external dependency)
│   ├── src/                 #   face detection + classifier building blocks (V1)
│   ├── src_v2/              #   V2 Quantum Haar-Wavelet recognizer (512-dim embedder)
│   ├── src_v3/              #   face_classifier.py — the 10-qubit face/non-face gate
│   ├── PQC/                 #   NTRU + AES-256-GCM metadata encryption (trimmed to runtime only)
│   └── models/             #   the 4 trained weights (qcnn_v2, qcnn_classifier, pca, scaler)
├── scripts/
│   ├── selfcheck.py         #   headless component check (run this first)
│   ├── enroll.py            #   register a face from the webcam
│   ├── test_camera.py       #   live camera + liveness preview (needs a display)
│   ├── run.sh / run.ps1     #   launch the kiosk
│   └── setup_jetson.sh, identify_board.sh   # for the Jetson path
├── tests/                   # pytest logic tests (no camera needed)
├── config.yaml              # all tunables — defaults set for a laptop webcam
└── requirements-app.txt     # pinned runtime dependencies
```

Generated on first run (git-ignored, safe to delete to reset): `vector_db/`, `events.db`,
`vendor/PQC/keys/`, `vendor/PQC/encrypted_metadata/`, `vendor/logs/`.

---


## Troubleshooting

- **`Cannot open webcam index 0`** — another app is using the camera, or it's a different index.
  Set `camera.webcam_index` in `config.yaml` (try 1), and on macOS grant camera permission to your
  terminal.
- **`cv2 … function not implemented` / no window** — you have a headless OpenCV build. Reinstall the
  GUI one: `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python && pip install opencv-contrib-python==4.11.0.86`.
- **Slow first recognition** — the very first call after boot warms the quantum circuits (one-time);
  steady state is ~1–1.5 s per recognition on a laptop CPU.
