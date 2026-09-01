"""
Face / Non-Face Classifier Utility for v3

Wraps the 10-qubit QCNN classifier (models/qcnn_classifier.pth) with
lazy-loading so it is shared across login, registration, and the standalone
main_v3 functions without being loaded multiple times.

Pipeline for a single face ROI:
    BGR face ROI
        → FacePreprocessor (16×16×3 flatten, 768-dim)
        → QuantumEncoder.reduce_dimensions()  (PCA: 768 → 10)
        → QuantumEncoder.encode_ry_rz()       (clip + scale to [0, π])
        → 10-qubit QCNN circuit                (25 trainable params)
        → softmax  →  [P(non-face), P(face)]

Returns:
    (is_face: bool, face_prob: float, non_face_prob: float)

Graceful fallback:
    If the classifier model or PCA/scaler files are missing the function
    returns (True, 1.0, 0.0) — i.e., the check is skipped transparently.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy-loaded singletons
_classifier = None
_encoder = None
_preprocessor = None


def _get_models():
    """
    Lazy-load the 10-qubit QCNN classifier, PCA encoder, and preprocessor.
    Returns (classifier, encoder, preprocessor) or (None, None, None) if unavailable.
    """
    global _classifier, _encoder, _preprocessor

    if _classifier is not None:
        return _classifier, _encoder, _preprocessor

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_path = os.path.join(project_root, "models", "qcnn_classifier.pth")
    pca_path = os.path.join(project_root, "models", "pca_model.pkl")
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")

    # Require all three files
    if not os.path.exists(classifier_path):
        print(f"  [FaceClassifier] qcnn_classifier.pth not found — skipping check")
        return None, None, None
    if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
        print(f"  [FaceClassifier] PCA/scaler models not found — skipping check")
        return None, None, None

    try:
        from src.qcnn_architecture import QCNN
        from src.quantum_encoding import QuantumEncoder
        from src.preprocessing import FacePreprocessor

        _classifier = QCNN()
        _classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
        _classifier.eval()

        _encoder = QuantumEncoder()      # loads pca_model.pkl + scaler.pkl
        _preprocessor = FacePreprocessor()

        print("  [FaceClassifier] 10-qubit QCNN classifier loaded successfully")
        return _classifier, _encoder, _preprocessor

    except Exception as e:
        print(f"  [FaceClassifier] Failed to load classifier: {e}")
        return None, None, None


def classify_face(face_roi_bgr: np.ndarray,
                  threshold: float = 0.50) -> tuple:
    """
    Classify a face ROI as Face or Non-Face using the 10-qubit QCNN.

    Args:
        face_roi_bgr : BGR numpy array — the cropped face region from OpenCV
        threshold    : Minimum face probability to accept (default 0.50)

    Returns:
        (is_face, face_prob, non_face_prob)
            is_face      — True if the crop is classified as a face
            face_prob    — P(face) in [0, 1]
            non_face_prob— P(non-face) in [0, 1]

    Graceful fallback:
        If the classifier is unavailable, returns (True, 1.0, 0.0) so the
        rest of the pipeline is not blocked.
    """
    classifier, encoder, preprocessor = _get_models()

    if classifier is None:
        # Model unavailable — pass through silently
        return True, 1.0, 0.0

    try:
        # Preprocess using the same pipeline as training (BGR input)
        processed = preprocessor.preprocess(face_roi_bgr)   # returns float [0,1]
        flat = processed.flatten()                           # 768-dim

        # PCA reduction (768 → 10) + StandardScaler
        reduced = encoder.reduce_dimensions(flat)            # (10,)

        # Normalize to [0, π] for RY/RZ encoding
        encoded = encoder.encode_ry_rz(reduced)              # (10,)

        input_tensor = torch.FloatTensor(encoded).unsqueeze(0)  # (1, 10)

        with torch.no_grad():
            logits = classifier(input_tensor)                # (1, 2)
            probs = torch.softmax(logits, dim=1)
            non_face_prob = probs[0, 0].item()               # class 0 = Non-Face
            face_prob = probs[0, 1].item()                   # class 1 = Face

        is_face = face_prob >= threshold
        return is_face, face_prob, non_face_prob

    except Exception as e:
        print(f"  [FaceClassifier] Classification error: {e}")
        # Fail-open: do not block the pipeline on unexpected errors
        return True, 1.0, 0.0
