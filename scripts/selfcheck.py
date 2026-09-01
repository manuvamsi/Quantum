"""
selfcheck.py — headless "does this machine have everything it needs?" check.

Run this FIRST on a new system (no camera, no face images required). It verifies the
install and that every model component loads and runs end-to-end on synthetic data:

  1. Python + platform + all required libraries import (with versions)
  2. The vendored model tree is present (weights + code)
  3. RecognitionEngine.load() succeeds  -> V2 recognizer + 10-qubit classifier + PQC-NTRU
     keys + ChromaDB all initialise
  4. The face/non-face classifier runs on a synthetic array  (quantum circuit executes)
  5. The V2 embedder returns a 512-dim vector on a synthetic face  (quantum circuit executes)
  6. (soft) The configured camera backend can open + read one frame

Exit code 0 = all HARD checks passed; 1 = something is missing/broken. The camera step is
a soft WARN (headless CI has no camera) — verify the camera live with test_camera.py / enroll.py.

    python scripts/selfcheck.py                # full check
    python scripts/selfcheck.py --no-camera    # skip the camera probe (pure headless)
"""

import os
import sys
import time
import platform
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root

OK = "PASS"
BAD = "FAIL"
WARN = "WARN"


def _p(status: str, label: str, detail: str = "") -> None:
    tail = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{tail}")


# Required libraries: import name -> pip name (for a helpful message on failure).
REQUIRED = [
    ("numpy", "numpy"),
    ("cv2", "opencv-contrib-python"),
    ("torch", "torch"),
    ("pennylane", "pennylane"),
    ("pennylane_lightning", "pennylane-lightning"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("chromadb", "chromadb"),
    ("cryptography", "cryptography"),
    ("yaml", "PyYAML"),
]


def check_libraries() -> bool:
    print("1) Required libraries")
    ok = True
    for mod, pip_name in REQUIRED:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            _p(OK, pip_name, ver)
        except Exception as e:  # noqa: BLE001
            _p(BAD, pip_name, f"cannot import '{mod}': {e}")
            ok = False
    return ok


def check_vendor(cfg) -> bool:
    from app.config import qfr_path
    print("2) Vendored model tree")
    root = qfr_path(cfg)
    needed = [
        "models/qcnn_v2_trained.pth",
        "models/qcnn_classifier.pth",
        "models/pca_model.pkl",
        "models/scaler.pkl",
        "src_v2/recognition_v2.py",
        "src_v3/face_classifier.py",
        "PQC/metadata_storage.py",
        "PQC/ntru/keygen.py",
    ]
    ok = True
    _p(OK, "qfr_path resolves to", root)
    for rel in needed:
        if os.path.exists(os.path.join(root, rel)):
            _p(OK, rel)
        else:
            _p(BAD, rel, "MISSING")
            ok = False
    return ok


def check_engine(cfg):
    from app.recognition.engine import RecognitionEngine
    print("3) Load engine (model + classifier + PQC + ChromaDB)")
    eng = RecognitionEngine(cfg)
    t = time.time()
    eng.load()
    _p(OK, "RecognitionEngine.load()", f"{time.time() - t:.1f}s")
    _p(OK, "ChromaDB collection", f"{eng.count()} enrolled embeddings")
    _p(OK, "classifier gate", "enabled" if eng._classify is not None else "disabled")
    return eng


def check_circuits(eng) -> bool:
    import numpy as np
    print("4) Run the quantum circuits on synthetic data")
    ok = True
    # Face/non-face classifier (10-qubit) — should return a probability without raising.
    try:
        if eng._classify is not None:
            is_face, fp = eng._is_face(np.zeros((96, 96, 3), np.uint8))
            _p(OK, "classifier ran", f"is_face={is_face} p={fp:.3f}")
        else:
            _p(WARN, "classifier disabled in config", "model.classify_enabled=false")
    except Exception as e:  # noqa: BLE001
        _p(BAD, "classifier", str(e))
        ok = False
    # V2 embedder (8-qubit) — should return a 512-dim vector.
    try:
        emb = eng._embed_roi(np.full((80, 80, 3), 127, np.uint8))
        dim = int(getattr(emb, "shape", [0])[0])
        if dim == int(eng.cfg.get("model", {}).get("embedding_dim", 512)):
            _p(OK, "V2 embedder", f"{dim}-dim vector")
        else:
            _p(BAD, "V2 embedder", f"unexpected dim {dim}")
            ok = False
    except Exception as e:  # noqa: BLE001
        _p(BAD, "V2 embedder", str(e))
        ok = False
    return ok


def check_camera(cfg) -> bool:
    from app.camera import get_camera
    backend = cfg.get("camera", {}).get("backend", "auto")
    print(f"5) Camera probe (backend='{backend}')  [soft]")
    try:
        with get_camera(cfg) as cam:
            rgb, _ = cam.read()
            if rgb is not None:
                h, w = rgb.shape[:2]
                _p(OK, "camera opened + read a frame", f"{w}x{h}")
                return True
            _p(WARN, "camera opened but read no frame", "check the device")
            return True
    except Exception as e:  # noqa: BLE001
        _p(WARN, "no camera available", f"{e} — fine for headless; test live later")
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-camera", action="store_true", help="skip the camera probe")
    args = ap.parse_args()

    print("=" * 64)
    print("ReQAgnIze — Testing_App_deployment self-check")
    print(f"  Python {platform.python_version()}  |  {platform.system()} {platform.machine()}")
    print("=" * 64)

    hard_ok = True

    if not check_libraries():
        print("\nRESULT: FAIL — install requirements first:")
        print("  pip install -r requirements-app.txt")
        return 1

    from app.config import load_config
    cfg = load_config()

    hard_ok &= check_vendor(cfg)
    if not hard_ok:
        print("\nRESULT: FAIL — the vendored model tree is incomplete.")
        return 1

    try:
        eng = check_engine(cfg)
    except Exception as e:  # noqa: BLE001
        _p(BAD, "engine failed to load", str(e))
        print("\nRESULT: FAIL — engine did not load.")
        return 1

    hard_ok &= check_circuits(eng)

    if not args.no_camera:
        check_camera(cfg)  # soft — never flips hard_ok

    print("=" * 64)
    if hard_ok:
        print("RESULT: PASS — all components load and run on this machine.")
        print("Next: 'python -m pytest tests -q', then enroll live with")
        print("      'python scripts/enroll.py --name \"Your Name\"' and run 'python -m app.main'.")
        return 0
    print("RESULT: FAIL — see the [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
