"""
Circuit B: Mid-Frequency Feature Extractor
Captures local textures, edges, and feature boundaries

Qubits: 6
Parameters: 14
Input: PCA features 2-7 (mid variance components, overlapping with A and C)
Output: 6 Pauli-Z expectations
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS_B = 6
N_PARAMS_B = 14

dev_b = qml.device("default.qubit", wires=N_QUBITS_B)


@qml.qnode(dev_b, interface="torch", diff_method="finite-diff")
def circuit_b_mid_frequency(inputs, weights):
    """
    Mid-frequency circuit for local textures

    Args:
        inputs: [6] PCA features (indices 2-7, mid variance)
        weights: [14] trainable parameters

    Returns:
        [6] Pauli-Z expectations on all qubits

    Architecture:
        - Encoding: RY + RZ for richer representation
        - Layer 1: IsingXX for X-basis correlations (texture edges)
        - Layer 2: IsingZZ for local correlations + CNOT cross-talk
        - Layer 3: U3 gates for maximum expressivity
        - Layer 4: CRX for final mixing
    """
    # === ENCODING: Mixed RY/RZ for texture capture ===
    for i in range(3):
        qml.RY(inputs[i], wires=i)
    for i in range(3, 6):
        qml.RZ(inputs[i], wires=i)

    # === LAYER 1: IsingXX for edge detection (X-basis correlations) ===
    qml.IsingXX(weights[0], wires=[0, 1])
    qml.IsingXX(weights[1], wires=[2, 3])
    qml.IsingXX(weights[2], wires=[4, 5])

    # === LAYER 2: IsingZZ for local correlations ===
    qml.IsingZZ(weights[3], wires=[0, 2])
    qml.IsingZZ(weights[4], wires=[1, 3])
    qml.IsingZZ(weights[5], wires=[2, 4])

    # Cross-talk between groups
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[3, 4])

    # === LAYER 3: U3 gates for maximum expressivity ===
    qml.U3(weights[6], weights[7], weights[8], wires=1)
    qml.U3(weights[9], weights[10], weights[11], wires=3)

    # === LAYER 4: Final CRX mixing ===
    qml.CRX(weights[12], wires=[0, 3])
    qml.CRX(weights[13], wires=[2, 5])

    # === MEASUREMENT ===
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_B)]


class QuantumCircuitB(nn.Module):
    """
    Quantum layer for mid-frequency features (local textures)

    Input: (batch, 6) - PCA features 2-7
    Output: (batch, 6) - quantum expectation values
    """

    def __init__(self, n_params=N_PARAMS_B):
        super().__init__()
        self.n_qubits = N_QUBITS_B
        self.n_params = n_params
        self.weights = nn.Parameter(torch.randn(n_params, dtype=torch.float32) * 0.05)

    def forward(self, x):
        """
        Forward pass for batch of inputs

        Args:
            x: (batch_size, 6) tensor

        Returns:
            (batch_size, 6) tensor
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            out = circuit_b_mid_frequency(x[i], self.weights)
            out_tensor = torch.stack(out)
            outputs.append(out_tensor)

        result = torch.stack(outputs).float()
        return result


if __name__ == "__main__":
    print("Testing Circuit B (Mid-Frequency)...")
    print("=" * 50)

    circuit = QuantumCircuitB()
    test_input = torch.randn(2, 6)

    print(f"Input shape: {test_input.shape}")
    print(f"Parameters: {circuit.n_params}")

    with torch.no_grad():
        output = circuit(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0].tolist()}")

    print("Circuit B test complete!")
