"""
Ensemble QCNN for Face Recognition

Combines 3 specialized quantum circuits:
- Circuit A: Low-frequency (global face structure)
- Circuit B: Mid-frequency (local textures)
- Circuit C: High-frequency (fine details)

Total quantum params: 34 (vs 45 for single QCNN)
Output: 512-dim L2-normalized embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit_a import QuantumCircuitA
from .circuit_b import QuantumCircuitB
from .circuit_c import QuantumCircuitC
from .fusion_layer import FusionLayer


class EnsembleQCNN(nn.Module):
    """
    Ensemble Quantum CNN for Face Recognition

    Architecture:
        Input: (batch, 10) PCA features
        ↓
        Feature Split:
            Circuit A: features[0:6]  → 6 qubits, 12 params → 6 outputs
            Circuit B: features[2:8]  → 6 qubits, 14 params → 6 outputs
            Circuit C: features[6:10] → 4 qubits, 8 params  → 4 outputs
        ↓
        Fusion Layer: 16 → 192 → 256 → 512
        ↓
        Output: (batch, 512) L2-normalized embeddings

    Total quantum parameters: 34
    Total classical parameters: ~200K (fusion layer)
    """

    def __init__(self, num_classes=None, dropout_rate=0.3):
        super().__init__()

        # Quantum circuits
        self.circuit_a = QuantumCircuitA(n_params=12)  # Low-frequency
        self.circuit_b = QuantumCircuitB(n_params=14)  # Mid-frequency
        self.circuit_c = QuantumCircuitC(n_params=8)   # High-frequency

        # Fusion layer
        self.fusion = FusionLayer(dropout_rate=dropout_rate)

        # Optional classifier head for training with softmax
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_embeddings=False):
        """
        Forward pass

        Args:
            x: (batch, 10) PCA features
            return_embeddings: If True, return embeddings instead of logits

        Returns:
            (batch, 512) embeddings or (batch, num_classes) logits
        """
        # Split features for each circuit (with overlap)
        inp_a = x[:, 0:6]    # Low frequency: PCA 0-5
        inp_b = x[:, 2:8]    # Mid frequency: PCA 2-7
        inp_c = x[:, 6:10]   # High frequency: PCA 6-9

        # Run quantum circuits
        out_a = self.circuit_a(inp_a)  # (batch, 6)
        out_b = self.circuit_b(inp_b)  # (batch, 6)
        out_c = self.circuit_c(inp_c)  # (batch, 4)

        # Fuse outputs
        embeddings = self.fusion(out_a, out_b, out_c)  # (batch, 512)

        if return_embeddings or self.num_classes is None:
            return embeddings

        # Classification (for training)
        logits = self.classifier(embeddings)
        return logits

    def extract_features(self, x):
        """
        Extract 512-dim L2-normalized embeddings for recognition

        Args:
            x: (batch, 10) PCA features

        Returns:
            (batch, 512) L2-normalized feature embeddings
        """
        embeddings = self.forward(x, return_embeddings=True)
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_quantum_params_count(self):
        """Return total quantum parameters"""
        return (self.circuit_a.n_params +
                self.circuit_b.n_params +
                self.circuit_c.n_params)

    def get_circuit_outputs(self, x):
        """
        Get raw outputs from each circuit (for debugging/analysis)

        Args:
            x: (batch, 10) PCA features

        Returns:
            dict with outputs from each circuit
        """
        inp_a = x[:, 0:6]
        inp_b = x[:, 2:8]
        inp_c = x[:, 6:10]

        with torch.no_grad():
            out_a = self.circuit_a(inp_a)
            out_b = self.circuit_b(inp_b)
            out_c = self.circuit_c(inp_c)

        return {
            'circuit_a': out_a,
            'circuit_b': out_b,
            'circuit_c': out_c,
            'attention_weights': self.fusion.get_attention_weights()
        }


if __name__ == "__main__":
    print("Testing Ensemble QCNN...")
    print("=" * 60)

    # Create model
    model = EnsembleQCNN(num_classes=5)

    # Test input (simulated PCA features)
    batch_size = 4
    test_input = torch.randn(batch_size, 10)

    print(f"\n1. Model Architecture:")
    print(f"   Quantum params: {model.get_quantum_params_count()}")
    print(f"   - Circuit A: {model.circuit_a.n_params} params")
    print(f"   - Circuit B: {model.circuit_b.n_params} params")
    print(f"   - Circuit C: {model.circuit_c.n_params} params")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params}")

    print(f"\n2. Forward Pass:")
    print(f"   Input shape: {test_input.shape}")

    with torch.no_grad():
        # Get embeddings
        embeddings = model.extract_features(test_input)
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Embedding norm: {torch.norm(embeddings[0]).item():.4f} (should be 1.0)")

        # Get logits
        logits = model(test_input)
        print(f"   Logits shape: {logits.shape}")

        # Get circuit outputs
        outputs = model.get_circuit_outputs(test_input)
        print(f"\n3. Circuit Outputs:")
        print(f"   Circuit A: {outputs['circuit_a'].shape}")
        print(f"   Circuit B: {outputs['circuit_b'].shape}")
        print(f"   Circuit C: {outputs['circuit_c'].shape}")
        print(f"   Attention weights: {outputs['attention_weights'].tolist()}")

    # Test embedding similarity
    print(f"\n4. Embedding Quality (random inputs):")
    with torch.no_grad():
        emb = model.extract_features(test_input)
        sim_matrix = torch.mm(emb, emb.t())
        print(f"   Self-similarity diagonal: {torch.diag(sim_matrix).tolist()}")
        mask = ~torch.eye(batch_size, dtype=bool)
        off_diag = sim_matrix[mask]
        print(f"   Off-diagonal mean: {off_diag.mean().item():.4f}")
        print(f"   Off-diagonal std: {off_diag.std().item():.4f}")

    print("\n" + "=" * 60)
    print("Ensemble QCNN test complete!")
