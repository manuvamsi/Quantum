"""
Ensemble QCNN Module

Provides an ensemble of 3 specialized quantum circuits for face recognition:
- Circuit A: Low-frequency features (global face structure)
- Circuit B: Mid-frequency features (local textures)
- Circuit C: High-frequency features (fine details)

Usage:
    from src.ensemble import EnsembleQCNN

    model = EnsembleQCNN()
    embeddings = model.extract_features(pca_features)  # (batch, 512)
"""

from .circuit_a import QuantumCircuitA, circuit_a_low_frequency
from .circuit_b import QuantumCircuitB, circuit_b_mid_frequency
from .circuit_c import QuantumCircuitC, circuit_c_high_frequency
from .fusion_layer import FusionLayer, SimpleFusionLayer
from .ensemble_model import EnsembleQCNN

__all__ = [
    'EnsembleQCNN',
    'QuantumCircuitA',
    'QuantumCircuitB',
    'QuantumCircuitC',
    'FusionLayer',
    'SimpleFusionLayer',
    'circuit_a_low_frequency',
    'circuit_b_mid_frequency',
    'circuit_c_high_frequency',
]
