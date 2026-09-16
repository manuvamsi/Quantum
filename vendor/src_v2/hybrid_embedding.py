"""
hybrid_embedding.py — Hybrid Quantum-Classical face embedding (1024-dim).

Why this exists: the pure quantum embedding (Haar wavelet stats -> 8-qubit QCNN)
is nearly input-independent — random noise scores ~0.95 cosine against an
enrolled face, so ANY face matches ANY person and the enroll duplicate-check
blocks every new registration. The circuit's 8 wavelet angles land in the same
narrow range for all natural images, so the 512-dim quantum signature cannot
separate people.

This extractor concatenates:
  - classical branch: InceptionResNetV1 (VGGFace2) 512-dim — real person
    discrimination (LFW-grade), provides the separation,
  - quantum branch: the existing 8-qubit QCNN + Haar-wavelet 512-dim — keeps
    the quantum circuit as part of the device's biometric signature.

Output: L2-normalized 1024-dim vector (each branch pre-normalized and weighted,
default 0.85 classical / 0.15 quantum).

Backend selection: model.embedding_backend = "hybrid" | "quantum" (config).
If facenet_pytorch is unavailable the extractor transparently falls back to the
pure quantum embedding (512-dim) so the app keeps running on minimal installs.
"""

import os
import threading

import cv2
import numpy as np

from src_v2.qcnn_recognition_v2 import QCNNEmbeddingExtractor

CLASSICAL_WEIGHT = 0.85
QUANTUM_WEIGHT = 0.15
FACENET_INPUT_SIZE = 160

_lock = threading.Lock()
_facenet_model = None
_facenet_failed = False


def _get_facenet():
    """Lazy-load InceptionResNetV1 (VGGFace2). Returns None if unavailable."""
    global _facenet_model, _facenet_failed
    if _facenet_failed:
        return None
    if _facenet_model is not None:
        return _facenet_model
    with _lock:
        if _facenet_model is not None:
            return _facenet_model
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2').eval()
            _facenet_model = model
        except Exception as e:
            print(f"[hybrid_embedding] facenet unavailable ({type(e).__name__}: {e}); "
                  f"falling back to quantum-only embedding")
            _facenet_failed = True
    return _facenet_model


class HybridEmbeddingExtractor:
    """1024-dim hybrid embedding: weighted concat of facenet(512) + QCNN(512)."""

    def __init__(self, seed: int = 42,
                 classical_weight: float = CLASSICAL_WEIGHT,
                 quantum_weight: float = QUANTUM_WEIGHT):
        self.seed = seed
        self.classical_weight = classical_weight
        self.quantum_weight = quantum_weight
        self.qcnn = QCNNEmbeddingExtractor(seed=seed)
        self.embedding_dim = 1024
        self.n_windows = self.qcnn.n_windows
        self._facenet_warned = False

    def _classical_embedding(self, image: np.ndarray) -> np.ndarray:
        """512-dim InceptionResNetV1 embedding of the face crop (RGB uint8)."""
        import torch
        model = _get_facenet()
        if model is None:
            return None
        img = cv2.resize(image, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE))
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()   # HWC->CHW, [0,255]
        x = (x - 127.5) / 128.0                                 # facenet standardization
        with torch.no_grad():
            e = model(x.unsqueeze(0))[0].cpu().numpy().astype(np.float32)
        n = np.linalg.norm(e)
        if n > 1e-8:
            e = e / n
        return e

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Hybrid embedding from an RGB uint8 face crop (any size)."""
        e_q = self.qcnn.extract_embedding(image)  # 512, L2-normalized
        e_c = self._classical_embedding(image)
        if e_c is None:
            if not self._facenet_warned:
                print("[hybrid_embedding] using quantum-only embedding (512-dim) — "
                      "person discrimination will be weak!")
                self._facenet_warned = True
            return e_q
        combined = np.concatenate([self.classical_weight * e_c,
                                   self.quantum_weight * e_q])
        n = np.linalg.norm(combined)
        if n > 1e-8:
            combined = combined / n
        return combined.astype(np.float32)

    def extract_embedding_batch(self, images) -> np.ndarray:
        return np.array([self.extract_embedding(img) for img in images])
