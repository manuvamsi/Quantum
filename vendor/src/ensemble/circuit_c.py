"""
Circuit C: High-Frequency Feature Extractor
Captures fine-grained details and micro-textures

Qubits: 4
Parameters: 8
Input: PCA features 6-9 (lowest variance components, fine details)
Output: 4 Pauli-Z expectations
"""

import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS_C = 4
N_PARAMS_C = 8

dev_c = qml.device("default.qubit", wires=N_QUBITS_C)


@qml.qnode(dev_c, interface="torch", diff_method="finite-diff")
def circuit_c_high_frequency(inputs, weights):
    """
    High-frequency circuit for fine details

    Args:
        inputs: [4] PCA features (indices 6-9, lowest variance)
        weights: [8] trainable parameters

    Returns:
        [4] Pauli-Z expectations on all qubits

    Architecture:
        - Encoding: Alternating RY/RZ for detail capture
        - Layer 1: Dense short-range entanglement (IsingXX + IsingYY)
        - Layer 2: Parameterized RY rotations
    """
    # === ENCODING: Alternating RY/RZ for fine detail capture ===
    qml.RY(inputs[0], wires=0)
    qml.RZ(inputs[1], wires=1)
    qml.RY(inputs[2], wires=2)
    qml.RZ(inputs[3], wires=3)

    # === LAYER 1: Dense short-range entanglement ===
    # IsingXX for adjacent pairs
    qml.IsingXX(weights[0], wires=[0, 1])
    qml.IsingXX(weights[1], wires=[2, 3])

    # IsingYY for cross connections
    qml.IsingYY(weights[2], wires=[1, 2])
    qml.IsingYY(weights[3], wires=[0, 3])

    # === LAYER 2: Parameterized rotations ===
    qml.RY(weights[4], wires=0)
    qml.RY(weights[5], wires=1)
    qml.RY(weights[6], wires=2)
    qml.RY(weights[7], wires=3)

    # === MEASUREMENT ===
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_C)]


class QuantumCircuitC(nn.Module):
    """
    Quantum layer for high-frequency features (fine details)

    Input: (batch, 4) - PCA features 6-9
    Output: (batch, 4) - quantum expectation values
    """

    def __init__(self, n_params=N_PARAMS_C):
        super().__init__()
        self.n_qubits = N_QUBITS_C
        self.n_params = n_params
        self.weights = nn.Parameter(torch.randn(n_params, dtype=torch.float32) * 0.05)

    def forward(self, x):
        """
        Forward pass for batch of inputs

        Args:
            x: (batch_size, 4) tensor

        Returns:
            (batch_size, 4) tensor
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            out = circuit_c_high_frequency(x[i], self.weights)
            out_tensor = torch.stack(out)
            outputs.append(out_tensor)

        result = torch.stack(outputs).float()
        return result


if __name__ == "__main__":
    print("Testing Circuit C (High-Frequency)...")
    print("=" * 50)

    circuit = QuantumCircuitC()
    test_input = torch.randn(2, 4)

    print(f"Input shape: {test_input.shape}")
    print(f"Parameters: {circuit.n_params}")

    with torch.no_grad():
        output = circuit(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0].tolist()}")

    print("Circuit C test complete!")
