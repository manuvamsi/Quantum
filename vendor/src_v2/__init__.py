"""
Quantum Face Recognition v2 - Quantum Haar Wavelet Feature Extraction

This version replaces bilinear pooling with Quantum Haar Wavelet Transform
for multi-resolution feature extraction before the 8-qubit QCNN processing.

Key differences from v1:
- Feature extraction: Haar wavelet (3-level decomposition) instead of bilinear pooling
- Database: Vector DB_v2/ (separate from v1)
- Better edge and texture preservation
- Multi-scale feature analysis

Usage:
    python main_v2.py           # CLI
    python live_recognition_v2.py  # Live camera
"""

__version__ = "2.0.0"
__author__ = "Quantum Face Recognition Team"

from .quantum_haar_wavelet import QuantumHaarWavelet, ClassicalHaarWavelet
