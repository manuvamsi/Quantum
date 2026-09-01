"""
Quantum Face Recognition v3 - Quantum Daubechies Wavelet Feature Extraction

This version replaces the classical Haar wavelet (v2) with a Quantum Daubechies
Wavelet Transform circuit integrated directly into the QCNN.

Key differences from v2:
- Feature extraction: Quantum Daubechies D4 wavelet (circuit layer, not classical)
- Wavelet runs inside the quantum circuit (fully-quantum pipeline)
- Database: Vector DB_v3/ (separate from v1 and v2)
- Same 34 trainable parameters in QCNN layers

Pipeline:
  Image -> 64x64 grayscale -> 64 8x8 windows
  -> Row-mean pooling -> 8 values [0, pi]
  -> RY angle encoding -> Quantum Daubechies Transform -> QCNN conv/pool
  -> 8 PauliZ measurements per window -> 512-dim embedding
"""

__version__ = "3.0.0"
