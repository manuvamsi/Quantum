"""
Fusion Layer for Ensemble QCNN
Combines outputs from 3 quantum circuits into 512-dim embeddings
Uses learnable attention weights to balance circuit contributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """
    Attention-based fusion layer that combines outputs from 3 quantum circuits

    Input:
        - out_a: (batch, 6) from Circuit A (low-frequency)
        - out_b: (batch, 6) from Circuit B (mid-frequency)
        - out_c: (batch, 4) from Circuit C (high-frequency)

    Output:
        - (batch, 512) L2-normalized embeddings

    Architecture:
        - Learnable attention weights for circuit balancing
        - Per-circuit projection to 64 dimensions
        - Concatenation: 64*3 = 192
        - Fusion MLP: 192 → 256 → 512
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # Learnable attention weights for each circuit
        # Allows model to learn which frequency band is most important
        self.circuit_attention = nn.Parameter(torch.ones(3))

        # Per-circuit projection to common dimension
        self.proj_a = nn.Linear(6, 64)   # Low-frequency
        self.proj_b = nn.Linear(6, 64)   # Mid-frequency
        self.proj_c = nn.Linear(4, 64)   # High-frequency

        # Fusion network: 192 → 512
        self.fusion = nn.Sequential(
            nn.Linear(192, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01)
        )

    def forward(self, out_a, out_b, out_c):
        """
        Fuse outputs from 3 quantum circuits

        Args:
            out_a: (batch, 6) from Circuit A (low-frequency)
            out_b: (batch, 6) from Circuit B (mid-frequency)
            out_c: (batch, 4) from Circuit C (high-frequency)

        Returns:
            (batch, 512) fused embeddings
        """
        # Compute softmax attention weights
        attn = F.softmax(self.circuit_attention, dim=0)

        # Project each circuit output with attention weighting
        proj_a = self.proj_a(out_a) * attn[0]
        proj_b = self.proj_b(out_b) * attn[1]
        proj_c = self.proj_c(out_c) * attn[2]

        # Concatenate projections
        combined = torch.cat([proj_a, proj_b, proj_c], dim=1)  # (batch, 192)

        # Fuse to final embedding
        embeddings = self.fusion(combined)  # (batch, 512)

        return embeddings

    def get_attention_weights(self):
        """Return the learned attention weights (for analysis)"""
        return F.softmax(self.circuit_attention, dim=0).detach()


class SimpleFusionLayer(nn.Module):
    """
    Simpler fusion layer without attention (for comparison)

    Just concatenates and projects to 512 dimensions
    """

    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # Direct fusion: 6 + 6 + 4 = 16 → 512
        self.fusion = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01)
        )

    def forward(self, out_a, out_b, out_c):
        """
        Simple concatenation and projection

        Args:
            out_a: (batch, 6) from Circuit A
            out_b: (batch, 6) from Circuit B
            out_c: (batch, 4) from Circuit C

        Returns:
            (batch, 512) fused embeddings
        """
        combined = torch.cat([out_a, out_b, out_c], dim=1)  # (batch, 16)
        embeddings = self.fusion(combined)  # (batch, 512)
        return embeddings


if __name__ == "__main__":
    print("Testing Fusion Layers...")
    print("=" * 50)

    # Test inputs (simulated quantum outputs)
    batch_size = 4
    out_a = torch.randn(batch_size, 6)
    out_b = torch.randn(batch_size, 6)
    out_c = torch.randn(batch_size, 4)

    # Test attention-based fusion
    print("\n1. Attention-Based Fusion Layer:")
    fusion = FusionLayer()
    embeddings = fusion(out_a, out_b, out_c)
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Attention weights: {fusion.get_attention_weights().tolist()}")

    # Test simple fusion
    print("\n2. Simple Fusion Layer:")
    simple_fusion = SimpleFusionLayer()
    embeddings_simple = simple_fusion(out_a, out_b, out_c)
    print(f"   Output shape: {embeddings_simple.shape}")

    # Count parameters
    print("\n3. Parameter Counts:")
    print(f"   Attention Fusion: {sum(p.numel() for p in fusion.parameters())}")
    print(f"   Simple Fusion: {sum(p.numel() for p in simple_fusion.parameters())}")

    print("\nFusion layer test complete!")
