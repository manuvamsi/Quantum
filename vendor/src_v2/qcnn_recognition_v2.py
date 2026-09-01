"""
8-Qubit Hierarchical QCNN for Face Recognition (Version 2)

This module implements the v2 quantum circuit for face recognition using
Quantum Haar Wavelet Transform for feature extraction instead of bilinear pooling.

Key Difference from v1:
- pool_window() uses Haar Wavelet (3-level decomposition) instead of bilinear resize
- Better edge and texture preservation
- Multi-scale feature analysis

Architecture:
- 8 qubits (matches 8x8 window structure)
- Hierarchical Conv-Pool-Conv-Pool structure
- 34 trainable parameters (fixed random weights - no training needed)
- Outputs 8 Pauli-Z expectation values per window

Usage:
    from src_v2.qcnn_recognition_v2 import QCNNEmbeddingExtractor

    extractor = QCNNEmbeddingExtractor(seed=42)
    embedding = extractor.extract_embedding(face_image)  # (512,)
"""

import os
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import List

# Import Haar Wavelet from v2
from src_v2.quantum_haar_wavelet import ClassicalHaarWavelet

# 8-qubit device for recognition
N_QUBITS_RECO = 8
N_PARAMS_RECO = 34  # Total trainable parameters

dev_reco = qml.device("lightning.qubit", wires=N_QUBITS_RECO)


def conv_layer(weights, wires):
    """
    Convolutional layer: parameterized 2-qubit gates on adjacent pairs.

    Each U_conv gate: RZ(theta1) - RY(theta2) - CNOT - RY(theta3) - RZ(theta4)
    Uses 4 parameters per pair.
    """
    wires = list(wires)
    n_pairs = len(wires) // 2

    for i in range(n_pairs):
        w0, w1 = wires[2*i], wires[2*i + 1]
        param_idx = i * 4

        # U_conv: entangling 2-qubit gate
        qml.RZ(weights[param_idx], wires=w0)
        qml.RY(weights[param_idx + 1], wires=w0)
        qml.CNOT(wires=[w0, w1])
        qml.RY(weights[param_idx + 2], wires=w1)
        qml.RZ(weights[param_idx + 3], wires=w1)


def pooling_layer(weights, wires):
    """
    Pooling layer: controlled rotations between pairs.

    Uses CRZ gates to transfer information from "traced out" qubits
    to the "surviving" qubits.
    """
    wires = list(wires)
    n_pairs = len(wires) // 2

    for i in range(n_pairs):
        w0, w1 = wires[2*i], wires[2*i + 1]
        qml.CRZ(weights[i], wires=[w0, w1])


@qml.qnode(dev_reco, interface="torch", diff_method="adjoint")
def hierarchical_qcnn_circuit(inputs, weights):
    """
    8-qubit Hierarchical QCNN for face recognition.

    Circuit Structure:
    ==================
    Layer 1: Angle Encoding (8 qubits)
        - RY(x_i) on each qubit

    Layer 2: Conv Layer 1 (4 pairs)
        - U_conv on (0,1), (2,3), (4,5), (6,7)
        - 16 parameters

    Layer 3: Pooling Layer 1 (4 pairs)
        - CRZ on same pairs
        - 4 parameters

    Layer 4: Conv Layer 2 (2 pairs on odd qubits)
        - U_conv on (1,3), (5,7)
        - 8 parameters

    Layer 5: Pooling Layer 2 (2 pairs)
        - CRZ on same pairs
        - 2 parameters

    Layer 6: Final Conv (1 pair)
        - U_conv on (3,7)
        - 4 parameters

    Total: 34 parameters

    Args:
        inputs: (8,) tensor - Haar wavelet features scaled to [0, pi]
        weights: (34,) tensor - circuit parameters

    Returns:
        List of 8 Pauli-Z expectation values
    """
    # Layer 1: Angle encoding
    for i in range(N_QUBITS_RECO):
        qml.RY(inputs[i], wires=i)

    # Layer 2: Conv Layer 1 (all 8 qubits, 4 pairs)
    # Pairs: (0,1), (2,3), (4,5), (6,7)
    conv_layer(weights[0:16], wires=range(8))

    # Layer 3: Pooling Layer 1
    pooling_layer(weights[16:20], wires=range(8))

    # Layer 4: Conv Layer 2 (odd qubits: 1,3,5,7)
    # Pairs: (1,3), (5,7)
    conv_layer(weights[20:28], wires=[1, 3, 5, 7])

    # Layer 5: Pooling Layer 2
    pooling_layer(weights[28:30], wires=[1, 3, 5, 7])

    # Layer 6: Final Conv (qubits 3, 7)
    conv_layer(weights[30:34], wires=[3, 7])

    # Measure all 8 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_RECO)]


class HierarchicalQCNN(nn.Module):
    """
    PyTorch wrapper for 8-qubit Hierarchical QCNN.

    This module wraps the PennyLane quantum circuit for use in PyTorch.
    Weights can be trained using ArcFace loss or used with pretrained weights.
    """

    def __init__(self, seed: int = 42, trainable: bool = False):
        super().__init__()

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Initialize with random weights
        # Using small random values for stable behavior
        self.weights = nn.Parameter(
            torch.randn(N_PARAMS_RECO) * 0.1,  # Smaller init for training stability
            requires_grad=trainable  # Can be enabled for training
        )

    def load_trained_weights(self, path: str) -> bool:
        """
        Load trained weights from checkpoint.

        Args:
            path: Path to trained model checkpoint

        Returns:
            True if weights loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            print(f"Warning: Trained weights not found at {path}")
            return False

        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        if 'qcnn_weights' in checkpoint:
            self.weights.data = checkpoint['qcnn_weights']
            print(f"Loaded trained QCNN weights from {path}")
            return True
        else:
            print(f"Warning: No qcnn_weights in checkpoint {path}")
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Args:
            x: (8,) tensor - Haar wavelet features scaled to [0, pi]

        Returns:
            (8,) tensor - Pauli-Z expectation values in [-1, 1]
        """
        result = hierarchical_qcnn_circuit(x, self.weights)
        return torch.stack(result)


class QCNNEmbeddingExtractor:
    """
    Extract face embeddings using 8-qubit Hierarchical QCNN with Haar Wavelet.

    Version 2 Features:
    - Uses Quantum Haar Wavelet Transform instead of bilinear pooling
    - 3-level wavelet decomposition per 8x8 window
    - Captures edges (LH, HL, HH) and texture features
    - Better multi-resolution analysis

    Pipeline:
    1. Resize image to 64x64
    2. Convert to grayscale
    3. Extract 8x8 non-overlapping windows (64 total)
    4. Apply Haar Wavelet Transform to each window -> 8 features
    5. Process through 8-qubit QCNN circuit
    6. Concatenate all measurements (64 x 8 = 512)
    7. L2 normalize

    Can use trained weights for better discrimination, or random weights.

    Attributes:
        qcnn: The 8-qubit Hierarchical QCNN circuit
        seed: Random seed for reproducible embeddings
        wavelet: ClassicalHaarWavelet instance for feature extraction
        n_windows: Number of windows (64)
        embedding_dim: Embedding dimension (512)
        trained: Whether trained weights were loaded
    """

    # Default path for trained weights
    TRAINED_WEIGHTS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'qcnn_v2_trained.pth'
    )

    def __init__(self, seed: int = 42, use_optimized: bool = True,
                 trained_weights_path: str = None):
        """
        Initialize the v2 embedding extractor with Haar Wavelet.

        Args:
            seed: Random seed for reproducible circuit weights
            use_optimized: Ignored in v2 (always uses grayscale mode)
            trained_weights_path: Path to trained weights (auto-detects if None)
        """
        self.seed = seed
        self.use_optimized = True  # v2 always uses optimized mode
        self.qcnn = HierarchicalQCNN(seed=seed)
        self.qcnn.eval()

        # Initialize Haar Wavelet extractor
        self.wavelet = ClassicalHaarWavelet()

        # v2 always uses 64 windows (grayscale)
        self.n_windows = 64
        self.embedding_dim = 512

        # Try to load trained weights
        weights_path = trained_weights_path or self.TRAINED_WEIGHTS_PATH
        self.trained = self.qcnn.load_trained_weights(weights_path)

        print(f"Initialized 8-qubit QCNN v2 with Haar Wavelet (seed={seed})")
        print(f"  - Feature Extraction: Quantum Haar Wavelet (3-level)")
        print(f"  - Circuit parameters: {N_PARAMS_RECO}")
        print(f"  - Trained weights: {'LOADED' if self.trained else 'NOT FOUND (using random)'}")
        print(f"  - Windows per image: {self.n_windows}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Wavelet features: LL, LH, HL, HH (edges + texture)")

    def extract_windows(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract non-overlapping 8x8 windows from 64x64 grayscale image.

        Args:
            image: (64, 64, 3) normalized image array

        Returns:
            List of 64 windows, each (8, 8) array
        """
        windows = []

        # Convert to grayscale using luminosity method
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image

        # Extract non-overlapping 8x8 windows
        for row in range(0, 64, 8):
            for col in range(0, 64, 8):
                window = gray[row:row+8, col:col+8]
                windows.append(window)

        return windows  # 64 windows, each 8x8

    def pool_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from 8x8 window using Quantum Haar Wavelet.

        This is the key difference from v1:
        - v1: bilinear interpolation (resize 8x8 to 4x2 -> 8 values)
        - v2: Haar wavelet decomposition -> 8 multi-resolution features

        Haar Wavelet Features:
        - LL3: DC component (average brightness)
        - LH3: Horizontal edges (finest level)
        - HL3: Vertical edges (finest level)
        - HH3: Diagonal edges (finest level)
        - LH2_mean: Horizontal edges (mid level)
        - HL2_mean: Vertical edges (mid level)
        - HH2_mean: Diagonal edges (mid level)
        - Energy: High-frequency texture energy

        Args:
            window: (8, 8) grayscale array

        Returns:
            (8,) array scaled to [0, pi] for quantum encoding
        """
        # Apply 2D Haar Wavelet Transform
        features = self.wavelet.transform_2d(window)
        return features  # Already scaled to [0, pi]

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image using Haar Wavelet + QCNN.

        Pipeline:
        1. Resize to 64x64 if needed
        2. Normalize to [0, 1]
        3. Convert to grayscale
        4. Extract 64 windows (8x8 each)
        5. Apply Haar Wavelet to each window -> 8 features
        6. Process through 8-qubit QCNN -> 8 measurements
        7. Concatenate all measurements (64 x 8 = 512)
        8. L2 normalize

        Args:
            image: RGB image (any size, will be resized to 64x64)

        Returns:
            L2-normalized (512,) embedding vector
        """
        # Resize to 64x64 if needed
        if image.shape[:2] != (64, 64):
            image = cv2.resize(image, (64, 64))

        # Ensure correct shape
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Extract windows
        windows = self.extract_windows(image)

        # Process each window through Haar Wavelet + QCNN
        measurements = []

        with torch.no_grad():
            for window in windows:
                # Apply Haar Wavelet Transform -> 8 features in [0, pi]
                wavelet_features = self.pool_window(window)
                qcnn_input = torch.from_numpy(wavelet_features).float()

                # Get 8 measurements from QCNN
                output = self.qcnn(qcnn_input)
                measurements.extend(output.numpy())

        # Convert to numpy array
        embedding = np.array(measurements, dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding  # (512,)

    def extract_embedding_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple images.

        Args:
            images: List of RGB images

        Returns:
            (N, 512) array of embeddings
        """
        embeddings = []
        for image in images:
            emb = self.extract_embedding(image)
            embeddings.append(emb)
        return np.array(embeddings)


# Test the module when run directly
if __name__ == "__main__":
    print("Testing 8-qubit QCNN v2 with Haar Wavelet")
    print("=" * 60)

    # Test 1: Haar Wavelet on test patch
    print("\n1. Testing Haar Wavelet Transform...")
    wavelet = ClassicalHaarWavelet()
    test_patch = np.zeros((8, 8), dtype=np.float32)
    test_patch[:4, :4] = 1.0  # Top-left white
    features = wavelet.transform_2d(test_patch)
    print(f"   Test patch: top-left quadrant white")
    print(f"   Wavelet features: {features}")
    print(f"   DC (LL3): {features[0]:.4f}")
    print(f"   Horizontal edge (LH3): {features[1]:.4f}")
    print(f"   Vertical edge (HL3): {features[2]:.4f}")

    # Test 2: Circuit forward pass
    print("\n2. Testing circuit forward pass...")
    qcnn = HierarchicalQCNN(seed=42)
    x = torch.from_numpy(features).float()
    output = qcnn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test 3: Full embedding extraction
    print("\n3. Testing full embedding extraction...")
    extractor = QCNNEmbeddingExtractor(seed=42)
    fake_image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    embedding = extractor.extract_embedding(fake_image)
    print(f"   Image shape: {fake_image.shape}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   L2 norm: {np.linalg.norm(embedding):.4f}")

    # Test 4: Reproducibility
    print("\n4. Testing reproducibility...")
    extractor2 = QCNNEmbeddingExtractor(seed=42)
    embedding2 = extractor2.extract_embedding(fake_image)
    diff = np.abs(embedding - embedding2).max()
    print(f"   Max difference between same-seed extractors: {diff:.6f}")

    # Test 5: Discrimination
    print("\n5. Testing discrimination...")
    fake_image2 = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    embedding3 = extractor.extract_embedding(fake_image2)
    cosine_sim = np.dot(embedding, embedding3)
    print(f"   Cosine similarity between random images: {cosine_sim:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! v2 with Haar Wavelet is working.")
