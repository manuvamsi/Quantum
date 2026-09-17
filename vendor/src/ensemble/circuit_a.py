"""
Circuit A: Low-Frequency Feature Extractor
Captures global face structure (symmetry, face shape, overall geometry)

Qubits: 6
Parameters: 12
Input: PCA features 0-5 (highest variance components)
Output: 6 Pauli-Z expectations
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS_A = 6
N_PARAMS_A = 12

dev_a = qml.device("default.qubit", wires=N_QUBITS_A)


@qml.qnode(dev_a, interface="torch", diff_method="finite-diff")
def circuit_a_low_frequency(inputs, weights):
    """
    Low-frequency circuit for global face structure

    Args:
        inputs: [6] PCA features (indices 0-5, highest variance)
        weights: [12] trainable parameters

    Returns:
        [6] Pauli-Z expectations on all qubits

    Architecture:
        - Layer 1: IsingYY symmetric pairs for face symmetry
        - Layer 2: CRZ ring for global connectivity
        - Layer 3: RY rotations for final encoding
    """
    # === ENCODING: RY for smooth global features ===
    for i in range(N_QUBITS_A):
        qml.RY(inputs[i], wires=i)

    # === LAYER 1: Symmetric pairs (face has left-right symmetry) ===
    qml.IsingYY(weights[0], wires=[0, 5])  # Outer symmetry (left-right)
    qml.IsingYY(weights[1], wires=[1, 4])  # Inner symmetry
    qml.IsingYY(weights[2], wires=[2, 3])  # Center pair

    # === LAYER 2: Global connectivity ring ===
    qml.CRZ(weights[3], wires=[0, 1])
    qml.CRZ(weights[4], wires=[1, 2])
    qml.CRZ(weights[5], wires=[2, 3])
    qml.CRZ(weights[6], wires=[3, 4])
    qml.CRZ(weights[7], wires=[4, 5])
    qml.CRZ(weights[8], wires=[5, 0])  # Close the ring

    # === LAYER 3: Final parameterized rotations ===
    qml.RY(weights[9], wires=0)
    qml.RY(weights[10], wires=2)
    qml.RY(weights[11], wires=4)

    # === MEASUREMENT ===
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_A)]


class QuantumCircuitA(nn.Module):
    """
    Quantum layer for low-frequency features (global face structure)

    Input: (batch, 6) - PCA features 0-5
    Output: (batch, 6) - quantum expectation values
    """

    def __init__(self, n_params=N_PARAMS_A):
        super().__init__()
        self.n_qubits = N_QUBITS_A
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
            out = circuit_a_low_frequency(x[i], self.weights)
            out_tensor = torch.stack(out)
            outputs.append(out_tensor)

        result = torch.stack(outputs).float()
        return result


if __name__ == "__main__":
    print("Testing Circuit A (Low-Frequency)...")
    print("=" * 50)

    circuit = QuantumCircuitA()
    test_input = torch.randn(2, 6)

    print(f"Input shape: {test_input.shape}")
    print(f"Parameters: {circuit.n_params}")

    with torch.no_grad():
        output = circuit(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0].tolist()}")

    print("Circuit A test complete!")
