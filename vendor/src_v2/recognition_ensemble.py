"""
recognition_ensemble.py — FaceRecognizer backed by the EnsembleQCNN embedder.

Reuses the proven v2 FaceRecognizer (ChromaDB init, recognize_with_voting, add_person,
save_database, get_stats) but replaces the embedder with the ensemble pipeline:

    face ROI (BGR)
      -> FacePreprocessor.preprocess           (16x16x3 -> flatten 768)
      -> QuantumEncoder.reduce_dimensions       (PCA 768 -> 10)
      -> QuantumEncoder.encode_ry_rz            (clip/scale to [0, pi])
      -> EnsembleQCNN.extract_features          (3 circuits + fusion -> 512-D, L2-normalized)

Weights resolve relative to CWD (the engine chdir's into the vendor dir before init):
    models/ensemble_qcnn.pth, models/pca_model.pkl, models/scaler.pkl
"""

import os
import numpy as np
import torch

from .recognition_v2 import FaceRecognizer as _V2FaceRecognizer


class FaceRecognizer(_V2FaceRecognizer):
    def __init__(self, db_dir=None, collection_name='face_embeddings_ensemble',
                 threshold=0.75, metric='cosine', seed: int = 42, **kwargs):
        # Reuse v2's ChromaDB / voting / add_person / stats machinery.
        super().__init__(db_dir=db_dir, collection_name=collection_name,
                         threshold=threshold, metric=metric, seed=seed)

        # Keep a separate pickle backup so we don't clobber the v2 one.
        self.pickle_path = os.path.join('models', 'recognition_db_ensemble.pkl')

        # Ensemble front-end (same PCA/preprocess path the face classifier uses).
        from src.quantum_encoding import QuantumEncoder
        from src.preprocessing import FacePreprocessor
        from src.ensemble.ensemble_model import EnsembleQCNN

        self.encoder = QuantumEncoder()               # loads models/pca_model.pkl + scaler.pkl
        self.ens_preprocessor = FacePreprocessor()

        weights = os.path.join('models', 'ensemble_qcnn.pth')
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        num_classes = state['classifier.weight'].shape[0] if 'classifier.weight' in state else None
        self.ensemble = EnsembleQCNN(num_classes=num_classes)
        self.ensemble.load_state_dict(state, strict=False)
        self.ensemble.eval()
        self.embedding_dim = 512
        print(f"Ensemble recognizer ready (EnsembleQCNN, 512-dim, "
              f"weights={'LOADED' if os.path.exists(weights) else 'MISSING'})")

    def embed_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Raw BGR face ROI -> 512-D L2-normalized ensemble embedding."""
        processed = self.ens_preprocessor.preprocess(roi_bgr)   # 16x16x3 float [0,1]
        flat = processed.flatten()                              # 768
        reduced = self.encoder.reduce_dimensions(flat)          # (10,)
        encoded = self.encoder.encode_ry_rz(reduced)            # (10,) in [0, pi]
        with torch.no_grad():
            v = self.ensemble.extract_features(
                torch.tensor(encoded, dtype=torch.float32).unsqueeze(0))
        return v.numpy()[0].astype(np.float32)                  # (512,)

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Alias so warm-up / any extract_embedding caller uses the ensemble pipeline."""
        return self.embed_roi(image)
