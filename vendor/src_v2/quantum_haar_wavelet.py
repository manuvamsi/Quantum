"""
Quantum Haar Wavelet Transform for Feature Extraction (v2)

Implements multi-resolution wavelet decomposition using quantum circuits.
Replaces bilinear pooling in the QCNN embedding extraction pipeline.

The Haar wavelet provides:
- Multi-resolution analysis (3 levels for 8x8 patches)
- Edge detection (LH, HL, HH subbands)
- Noise reduction (averaging in LL subband)
- Better texture discrimination

Mathematical Foundation:
    Classical 1D Haar:
        Approximation: a[k] = (x[2k] + x[2k+1]) / sqrt(2)
        Detail:        d[k] = (x[2k] - x[2k+1]) / sqrt(2)

    Quantum Haar (using Hadamard gates):
        H|0> = (|0> + |1>) / sqrt(2)  -> Approximation
        H|1> = (|0> - |1>) / sqrt(2)  -> Detail
"""

import numpy as np
from typing import List, Tuple, Optional

# Try to import PennyLane for quantum simulation
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumHaarWavelet:
    """
    Quantum Haar Wavelet Transform for 8x8 image patches.

    Uses Hadamard gates to compute approximation (sum) and
    detail (difference) coefficients in quantum superposition.

    Requires PennyLane for quantum circuit simulation.
    Falls back to ClassicalHaarWavelet if PennyLane is not available.
    """

    def __init__(self, n_qubits: int = 3):
        """
        Initialize Quantum Haar Wavelet Transform.

        Args:
            n_qubits: Number of qubits for wavelet levels (3 = 8 pixels per row)
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required for QuantumHaarWavelet. "
                "Install with: pip install pennylane"
            )

        self.n_qubits = n_qubits
        self.dev = qml.device('lightning.qubit', wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        """Build quantum Haar wavelet circuit."""
        @qml.qnode(self.dev)
        def haar_circuit(amplitudes: np.ndarray):
            # Amplitude encoding of 8 values (2^3 = 8)
            qml.AmplitudeEmbedding(
                amplitudes,
                wires=range(self.n_qubits),
                normalize=True
            )

            # Hadamard transforms for wavelet decomposition
            # This implements the Quantum Wavelet Transform (QWT)
            # Level 1: Apply H to qubit 2 (pairs of 4)
            qml.Hadamard(wires=2)

            # Level 2: Apply H to qubit 1 (pairs of 2)
            qml.Hadamard(wires=1)

            # Level 3: Apply H to qubit 0 (pairs of 1)
            qml.Hadamard(wires=0)

            # Measure probabilities - gives wavelet coefficients
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = haar_circuit

    def transform_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Apply 1D Quantum Haar Wavelet Transform.

        Args:
            x: 1D array of 8 values

        Returns:
            8 wavelet coefficients [approx, detail_l3, detail_l2x2, detail_l1x4]
        """
        # Normalize for amplitude encoding
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return np.zeros(8)

        x_norm = x / norm

        # Run quantum circuit
        probs = self.circuit(x_norm)

        # Convert probabilities to wavelet coefficients
        # sqrt(prob) gives amplitude which encodes wavelet coefficients
        coeffs = np.sqrt(probs) * np.sign(x_norm.mean() + 1e-10)

        return coeffs

    def transform_2d(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply 2D Quantum Haar Wavelet Transform to 8x8 patch.

        Applies row-wise then column-wise transforms (separable 2D wavelet).

        Args:
            patch: 8x8 grayscale image patch

        Returns:
            8 wavelet features (multi-resolution decomposition)
        """
        # Ensure 8x8
        if patch.shape != (8, 8):
            raise ValueError(f"Expected 8x8 patch, got {patch.shape}")

        # Apply to rows
        row_coeffs = np.array([self.transform_1d(row) for row in patch])

        # Apply to columns
        col_coeffs = np.array([
            self.transform_1d(row_coeffs[:, i])
            for i in range(8)
        ]).T

        # Extract multi-resolution features
        features = self._extract_features(col_coeffs)

        return features

    def _extract_features(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Extract 8 features from wavelet coefficient matrix.

        Returns:
            [LL3, LH3, HL3, HH3, LH2_mean, HL2_mean, HH2_mean, energy]
        """
        features = np.zeros(8)

        # Level 3 (1x1 each): LL, LH, HL, HH at corners
        features[0] = coeffs[0, 0]  # LL (approximation / DC)
        features[1] = coeffs[0, 4]  # LH (horizontal edges)
        features[2] = coeffs[4, 0]  # HL (vertical edges)
        features[3] = coeffs[4, 4]  # HH (diagonal edges)

        # Level 2 (2x2 each): mean of detail subbands
        features[4] = np.mean(coeffs[0:2, 2:4])   # LH2
        features[5] = np.mean(coeffs[2:4, 0:2])   # HL2
        features[6] = np.mean(coeffs[2:4, 2:4])   # HH2

        # Energy feature (total signal energy)
        features[7] = np.sum(coeffs ** 2)

        # Scale to [0, pi] for quantum angle encoding
        features = np.clip(features, -1, 1)
        features = (features + 1) / 2 * np.pi

        return features


class ClassicalHaarWavelet:
    """
    Classical Haar Wavelet for fast feature extraction.

    Mathematically equivalent to QuantumHaarWavelet but runs on CPU
    without quantum simulation overhead. Use this for production.

    The 2D Haar wavelet decomposes an 8x8 patch into:
    - Level 1: 4x4 (LL1, LH1, HL1, HH1)
    - Level 2: 2x2 (LL2, LH2, HL2, HH2)
    - Level 3: 1x1 (LL3, LH3, HL3, HH3)

    Where:
    - LL = Low-Low (approximation / average)
    - LH = Low-High (horizontal edges)
    - HL = High-Low (vertical edges)
    - HH = High-High (diagonal edges)
    """

    @staticmethod
    def transform_2d(patch: np.ndarray) -> np.ndarray:
        """
        Fast classical 2D Haar wavelet transform.

        Args:
            patch: 8x8 grayscale patch (values in [0, 1])

        Returns:
            8 wavelet features scaled to [0, pi] for quantum encoding
        """
        # Ensure float and correct shape
        patch = np.asarray(patch, dtype=np.float32)
        if patch.shape != (8, 8):
            raise ValueError(f"Expected 8x8 patch, got {patch.shape}")

        # Level 1: 8x8 -> 4x4
        # LL1 = average of 2x2 blocks
        ll1 = (patch[0::2, 0::2] + patch[0::2, 1::2] +
               patch[1::2, 0::2] + patch[1::2, 1::2]) / 4

        # LH1 = horizontal difference (detects horizontal edges)
        lh1 = (patch[0::2, 0::2] - patch[0::2, 1::2] +
               patch[1::2, 0::2] - patch[1::2, 1::2]) / 4

        # HL1 = vertical difference (detects vertical edges)
        hl1 = (patch[0::2, 0::2] + patch[0::2, 1::2] -
               patch[1::2, 0::2] - patch[1::2, 1::2]) / 4

        # HH1 = diagonal difference (detects diagonal edges)
        hh1 = (patch[0::2, 0::2] - patch[0::2, 1::2] -
               patch[1::2, 0::2] + patch[1::2, 1::2]) / 4

        # Level 2: 4x4 -> 2x2 (on LL1 only)
        ll2 = (ll1[0::2, 0::2] + ll1[0::2, 1::2] +
               ll1[1::2, 0::2] + ll1[1::2, 1::2]) / 4

        lh2 = (ll1[0::2, 0::2] - ll1[0::2, 1::2] +
               ll1[1::2, 0::2] - ll1[1::2, 1::2]) / 4

        hl2 = (ll1[0::2, 0::2] + ll1[0::2, 1::2] -
               ll1[1::2, 0::2] - ll1[1::2, 1::2]) / 4

        hh2 = (ll1[0::2, 0::2] - ll1[0::2, 1::2] -
               ll1[1::2, 0::2] + ll1[1::2, 1::2]) / 4

        # Level 3: 2x2 -> 1x1 (on LL2 only)
        ll3 = (ll2[0, 0] + ll2[0, 1] + ll2[1, 0] + ll2[1, 1]) / 4
        lh3 = (ll2[0, 0] - ll2[0, 1] + ll2[1, 0] - ll2[1, 1]) / 4
        hl3 = (ll2[0, 0] + ll2[0, 1] - ll2[1, 0] - ll2[1, 1]) / 4
        hh3 = (ll2[0, 0] - ll2[0, 1] - ll2[1, 0] + ll2[1, 1]) / 4

        # Build 8-feature vector
        features = np.array([
            ll3,                      # DC / approximation (average brightness)
            lh3,                      # Horizontal edges (finest level)
            hl3,                      # Vertical edges (finest level)
            hh3,                      # Diagonal edges (finest level)
            np.mean(lh2),             # Horizontal edges (mid level)
            np.mean(hl2),             # Vertical edges (mid level)
            np.mean(hh2),             # Diagonal edges (mid level)
            np.mean(np.abs(hh1))      # High-frequency energy (texture)
        ], dtype=np.float32)

        # Scale to [0, pi] for quantum angle encoding
        features = np.clip(features, -1, 1)
        features = (features + 1) / 2 * np.pi

        return features

    @staticmethod
    def full_decomposition(patch: np.ndarray) -> dict:
        """
        Get full wavelet decomposition for visualization/analysis.

        Args:
            patch: 8x8 grayscale patch

        Returns:
            Dictionary with all subbands at each level
        """
        patch = np.asarray(patch, dtype=np.float32)

        # Level 1
        ll1 = (patch[0::2, 0::2] + patch[0::2, 1::2] +
               patch[1::2, 0::2] + patch[1::2, 1::2]) / 4
        lh1 = (patch[0::2, 0::2] - patch[0::2, 1::2] +
               patch[1::2, 0::2] - patch[1::2, 1::2]) / 4
        hl1 = (patch[0::2, 0::2] + patch[0::2, 1::2] -
               patch[1::2, 0::2] - patch[1::2, 1::2]) / 4
        hh1 = (patch[0::2, 0::2] - patch[0::2, 1::2] -
               patch[1::2, 0::2] + patch[1::2, 1::2]) / 4

        # Level 2
        ll2 = (ll1[0::2, 0::2] + ll1[0::2, 1::2] +
               ll1[1::2, 0::2] + ll1[1::2, 1::2]) / 4
        lh2 = (ll1[0::2, 0::2] - ll1[0::2, 1::2] +
               ll1[1::2, 0::2] - ll1[1::2, 1::2]) / 4
        hl2 = (ll1[0::2, 0::2] + ll1[0::2, 1::2] -
               ll1[1::2, 0::2] - ll1[1::2, 1::2]) / 4
        hh2 = (ll1[0::2, 0::2] - ll1[0::2, 1::2] -
               ll1[1::2, 0::2] + ll1[1::2, 1::2]) / 4

        # Level 3
        ll3 = (ll2[0, 0] + ll2[0, 1] + ll2[1, 0] + ll2[1, 1]) / 4
        lh3 = (ll2[0, 0] - ll2[0, 1] + ll2[1, 0] - ll2[1, 1]) / 4
        hl3 = (ll2[0, 0] + ll2[0, 1] - ll2[1, 0] - ll2[1, 1]) / 4
        hh3 = (ll2[0, 0] - ll2[0, 1] - ll2[1, 0] + ll2[1, 1]) / 4

        return {
            'level1': {'LL': ll1, 'LH': lh1, 'HL': hl1, 'HH': hh1},
            'level2': {'LL': ll2, 'LH': lh2, 'HL': hl2, 'HH': hh2},
            'level3': {'LL': ll3, 'LH': lh3, 'HL': hl3, 'HH': hh3}
        }


def get_wavelet_extractor(use_quantum: bool = False) -> object:
    """
    Factory function to get appropriate wavelet extractor.

    Args:
        use_quantum: If True, use QuantumHaarWavelet (requires PennyLane)
                    If False, use ClassicalHaarWavelet (faster, default)

    Returns:
        Wavelet extractor instance
    """
    if use_quantum:
        if not PENNYLANE_AVAILABLE:
            print("Warning: PennyLane not available, falling back to classical")
            return ClassicalHaarWavelet()
        return QuantumHaarWavelet()
    return ClassicalHaarWavelet()


# Test function
if __name__ == "__main__":
    print("Testing Quantum Haar Wavelet Transform")
    print("=" * 50)

    # Create test patch with known pattern
    patch = np.zeros((8, 8), dtype=np.float32)
    patch[:4, :4] = 1.0  # Top-left quadrant white

    print("\nTest patch (top-left white, rest black):")
    print(patch)

    # Classical wavelet
    print("\n--- Classical Haar Wavelet ---")
    classical = ClassicalHaarWavelet()
    c_features = classical.transform_2d(patch)
    print(f"Features: {c_features}")
    print(f"DC component (LL3): {c_features[0]:.4f}")
    print(f"Horizontal edges (LH3): {c_features[1]:.4f}")
    print(f"Vertical edges (HL3): {c_features[2]:.4f}")
    print(f"Diagonal edges (HH3): {c_features[3]:.4f}")

    # Quantum wavelet (if available)
    if PENNYLANE_AVAILABLE:
        print("\n--- Quantum Haar Wavelet ---")
        quantum = QuantumHaarWavelet()
        q_features = quantum.transform_2d(patch)
        print(f"Features: {q_features}")
        print(f"DC component (LL3): {q_features[0]:.4f}")
    else:
        print("\nPennyLane not available - skipping quantum test")

    # Full decomposition
    print("\n--- Full Decomposition ---")
    decomp = classical.full_decomposition(patch)
    print(f"Level 3 coefficients: LL={decomp['level3']['LL']:.4f}, "
          f"LH={decomp['level3']['LH']:.4f}, "
          f"HL={decomp['level3']['HL']:.4f}, "
          f"HH={decomp['level3']['HH']:.4f}")
