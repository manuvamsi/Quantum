"""
engine.py — the edge recognition engine.

Wraps the existing V2 (Quantum Haar-Wavelet) recognition pipeline for on-device use,
adds PQC-NTRU encrypted identity, and keeps a device-local ChromaDB. It reuses the
proven pipeline from `quantum_face_recognition/web_api/recognition_service.py`
(detect -> preprocess -> embed -> recognize_with_voting) and adds enroll + PQC.

The heavy model is loaded ONCE (on power-on) and kept ; recognition runs only
after liveness passes. The V2 embedder already uses `lightning.qubit`(This is Quantum SDK simulation device name which is written in python) for speed.
"""

import os
import sys
import time
import threading
from typing import List, Optional

import cv2
import numpy as np

from app.config import abspath, qfr_path


class EngineError(Exception):
    """User-facing recognition/enrollment failure."""


class RecognitionEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg.get("model", {})
        self.threshold = float(m.get("recognition_threshold", 0.75))
        self.collection = m.get("chroma_collection", "face_embeddings_v2")
        self.vector_db_dir = abspath(cfg, m.get("vector_db_dir", "vector_db"))
        self.qfr = qfr_path(cfg)
        self.classify_enabled = bool(m.get("classify_enabled", True))
        self.classify_threshold = float(m.get("classify_threshold", 0.50))
        self.block_threshold = float(m.get("block_threshold", 0.90))  # enroll duplicate gate

        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._ready = False
        self.recognizer = None
        self.preprocessor = None
        self.detector = None
        self.meta = None
        self._classify = None         # 10-qubit face/non-face classifier (V1-style gate)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self):
        """Import + instantiate the heavy V2 model, PQC storage, and ChromaDB (once)."""
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            if self.qfr not in sys.path:
                sys.path.insert(0, self.qfr)

            # The V2 embedder auto-detects models/qcnn_v2_trained.pth D,
            # so load from the model dir; the ChromaDB path is passed explicitly.
            prev = os.getcwd()
            os.chdir(self.qfr)
            try:
                version = str(self.cfg.get("model", {}).get("version", "v2")).lower()
                if version == "ensemble":
                    from src_v2.recognition_ensemble import FaceRecognizer
                else:
                    from src_v2.recognition_v2 import FaceRecognizer
                from src_v2.preprocessing import FacePreprocessor
                from src.face_detection import FaceDetector
                from PQC.metadata_storage import get_metadata_storage
                from PQC.ntru import NTRUKeyGenerator

                # New device: ensure NTRU keys exist (generates on first run only).
                keys_dir = os.path.join(self.qfr, "PQC", "keys")
                if not (os.path.exists(os.path.join(keys_dir, "public_key.pkl"))
                        and os.path.exists(os.path.join(keys_dir, "private_key.pkl"))):
                    NTRUKeyGenerator().generate_and_save()

                self.recognizer = FaceRecognizer(
                    db_dir=self.vector_db_dir,
                    collection_name=self.collection,
                    threshold=self.threshold,
                )
                self.preprocessor = FacePreprocessor()
                self.detector = FaceDetector()
                self.meta = get_metadata_storage()

                # 10-qubit face/non-face classifier (same gate as V1). Fails open if the
                # classifier / PCA / scaler models are missing.
                if self.classify_enabled:
                    from src_v3.face_classifier import classify_face
                    self._classify = classify_face

                # Warm BOTH quantum circuits at power-on so the first real recognition is
               
                dummy = np.zeros((96, 96, 3), np.uint8)
                for _ in range(2):
                    try:
                        if self._classify is not None:
                            self._classify(dummy, self.classify_threshold)
                        self.recognizer.extract_embedding(np.zeros((64, 64, 3), np.uint8))
                    except Exception:
                        pass
            finally:
                os.chdir(prev)
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def count(self) -> int:
        if not self._ready:
            return 0
        try:
            return self.recognizer.collection.count()
        except Exception:
            return 0

    def list_users(self) -> List[str]:
        self.load()
        try:
            return self.recognizer.get_stats().get("people", [])
        except Exception:
            return []

    # ── core pipeline ─────────────────────────────────────────────────────────
    def _detect_roi(self, img_bgr: np.ndarray) -> np.ndarray:
        """Detect the largest face and return its BGR ROI (raises if none)i.e., largest face comes when a person is near to camera."""
        faces = self.detector.detect_faces(img_bgr)
        if not faces:
            raise EngineError("No face detected")
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return img_bgr[y:y + h, x:x + w]

    def _is_face(self, roi_bgr: np.ndarray):
        """10-qubit face/non-face gate (V1). Returns (is_face, face_prob); (True, 1.0) if disabled."""
        if self._classify is None:
            return True, 1.0
        is_face, face_prob, _ = self._classify(roi_bgr, self.classify_threshold)
        return is_face, face_prob

    def _embed_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return the 512-dim embedding for a face ROI. If the recognizer owns its own
        preprocessing (exposes embed_roi, e.g. the ensemble), use it; else the V2 64x64 path."""
        if hasattr(self.recognizer, "embed_roi"):
            return self.recognizer.embed_roi(roi_bgr)
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pre = self.preprocessor.preprocess(rgb, for_recognition=True)
        return self.recognizer.extract_embedding((pre * 255).astype(np.uint8))

    def _lookup_user_id(self, embedding: np.ndarray) -> Optional[str]:
        try:
            res = self.recognizer.collection.query(
                query_embeddings=[embedding.astype(float).tolist()],
                n_results=1, include=["metadatas"],
            )
            if res.get("metadatas") and res["metadatas"][0]:
                return res["metadatas"][0][0].get("user_id")
        except Exception:
            pass
        return None

    # ── operations ────────────────────────────────────────────────────────────
    def enroll(self, name: str, images_bgr: List[np.ndarray], phone: str = None, age=None) -> dict:
        """Register a person from 1+ face frames. Embeddings -> ChromaDB; PII -> NTRU."""
        self.load()
        name = (name or "").strip()
        if len(name) < 2:
            raise EngineError("Name must be at least 2 characters")
        if not images_bgr:
            raise EngineError("At least one face image is required")

        with self._infer_lock:
            embeddings = []
            for i, img in enumerate(images_bgr):
                try:
                    roi = self._detect_roi(img)
                except EngineError as e:
                    raise EngineError(f"Image {i + 1}: {e}")
                is_face, fp = self._is_face(roi)
                if not is_face:
                    raise EngineError(f"Image {i + 1}: not a face (p={fp:.2f})")
                embeddings.append(self._embed_roi(roi))

            # Block only if VERY close to an existing user (direct top-1 cosine similarity vs a
            # dedicated block threshold, so a genuinely different face is not rejected as a duplicate).
            if self.recognizer.collection.count() > 0:
                for emb in embeddings:
                    res = self.recognizer.collection.query(
                        query_embeddings=[emb.astype(float).tolist()],
                        n_results=min(5, self.recognizer.collection.count()),
                        include=["distances", "metadatas"],
                    )
                    dists = res.get("distances") or [[]]
                    if dists[0]:
                        top_sim = 1.0 - dists[0][0]
                        if top_sim > self.block_threshold:
                            match = (res.get("metadatas") or [[{}]])[0][0].get("name", "Unknown")
                            raise EngineError(f"Already registered as '{match}' (sim {top_sim:.2f})")

            user_id = name.lower().replace(" ", "_")
            self.recognizer.add_person(name, embeddings, phone=phone, age=age, user_id=user_id)
            self.recognizer.save_database()
            # Encrypt full PII with NTRU (name/phone/age)
            self.meta.store_metadata(user_id, {
                "name": name, "phone": phone or "", "age": str(age) if age is not None else "",
            })

        return {"success": True, "user_id": user_id, "name": name,
                "embeddings": len(embeddings)}

    def recognize(self, img_bgr: np.ndarray) -> dict:
        """Authenticate one captured frame. Returns access decision + decrypted identity."""
        self.load()
        t0 = time.time()
        with self._infer_lock:
            roi = self._detect_roi(img_bgr)
            is_face, fp = self._is_face(roi)          # V1-style gate, after liveness
            if not is_face:
                return {"allowed": False, "user": None, "score": 0.0, "votes": 0,
                        "message": f"Not a face (p={fp:.2f})",
                        "latency_sec": round(time.time() - t0, 3)}
            emb = self._embed_roi(roi)
            user, score, votes = self.recognizer.recognize_with_voting(emb, n_results=5)
        score = float(score)

        if user != "Unknown" and score >= self.threshold:
            user_id = self._lookup_user_id(emb)
            meta = (self.meta.load_metadata(user_id) if user_id else None) or {"name": user}
            return {"allowed": True, "user": user, "score": score, "votes": int(votes),
                    "metadata": meta, "latency_sec": round(time.time() - t0, 3)}
        return {"allowed": False, "user": None, "score": score, "votes": int(votes),
                "message": "Not recognized", "latency_sec": round(time.time() - t0, 3)}
