"""
Quantum Convolutional Neural Network
Based on HQCNN paper architecture
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

N_QUBITS = 10
# lightning.qubit (C++ state-vector) — much faster inference than default.qubit for the
# face/non-face gate; identical results. Falls back to default.qubit if lightning is absent.
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="finite-diff") #other diff_method is parameter-shift, finite-diff is faster for this circuit
def qcnn_circuit(inputs, weights):
    """
    Full QCNN circuit with encoding, quanvolution, pooling
    
    Args:
        inputs: [10] PCA-reduced features
        weights: [25] trainable parameters
    
    Returns:
        Pauli-Z expectation on q0
    """
    
    #  ENCODING LAYER (RYZ)
    for i in range(5):
        qml.RY(inputs[i], wires=i)
    for i in range(5, 10):
        qml.RZ(inputs[i], wires=i)
    
    # LAYER 1 
    # IsingYY (intra-channel entanglement)
    qml.IsingYY(weights[0], wires=[0, 1])  # R-channel
    qml.IsingYY(weights[1], wires=[2, 3])  # G-channel
    qml.IsingYY(weights[2], wires=[4, 5])  # B-channel
    
    # Cross-channel CNOT
    qml.CNOT(wires=[1, 2])  # R→G
    qml.CNOT(wires=[3, 4])  # G→B
    
    # IsingZZ (phase structure)
    qml.IsingZZ(weights[3], wires=[0, 1])
    qml.IsingZZ(weights[4], wires=[2, 3])
    qml.IsingZZ(weights[5], wires=[4, 5])
    
    # ===== UOA OPERATOR (CRx gates) =====
    qml.CRX(weights[6], wires=[0, 2])  # R→G
    qml.CRX(weights[7], wires=[2, 4])  # G→B
    qml.CRX(weights[8], wires=[4, 0])  # B→R (ring)
    
    # ===== POOLING LAYER 1 (CP gates) =====
    # Keep q0 (R main), q2 (G main), q4 (B main)
    qml.PhaseShift(weights[9], wires=0)
    qml.PhaseShift(weights[10], wires=2)
    qml.PhaseShift(weights[11], wires=4)
    
    # ===== QUANVOLUTIONAL LAYER 2 =====
    # U3 gates on main qubits
    qml.U3(weights[12], weights[13], weights[14], wires=0)
    qml.U3(weights[15], weights[16], weights[17], wires=2)
    qml.U3(weights[18], weights[19], weights[20], wires=4)
    
    # IsingYY + IsingZZ between main qubits
    qml.IsingYY(weights[21], wires=[0, 2])
    qml.IsingZZ(weights[22], wires=[2, 4])
    
    # ===== POOLING LAYER 2 =====
    # Compress all to q0
    qml.PhaseShift(weights[23], wires=0)
    qml.CNOT(wires=[2, 0])
    qml.PhaseShift(weights[24], wires=0)
    qml.CNOT(wires=[4, 0])
    
    # ===== MEASUREMENT =====
    return qml.expval(qml.PauliZ(0))


class QuantumLayer(nn.Module):
    """Wrapper for quantum circuit with proper batch handling"""
    
    def __init__(self, n_qubits=N_QUBITS, n_params=25):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.weights = nn.Parameter(torch.randn(n_params, dtype=torch.float32) * 0.01)
    
    def forward(self, x):
        """
        Forward pass for batch of inputs
        x: (batch_size, 10) tensor
        Returns: (batch_size, 1) tensor
        """
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Process each sample individually
            out = qcnn_circuit(x[i], self.weights)
            outputs.append(out)
        
        # Stack and convert to float32 (PennyLane returns float64)
        result = torch.stack(outputs).float().unsqueeze(-1)  # (batch, 1)
        return result


class QCNN(nn.Module):
    """
    Hybrid Quantum-Classical CNN
    """
    
    def __init__(self):
        super().__init__()
        
        # Quantum layer (25 trainable parameters)
        self.qlayer = QuantumLayer(n_qubits=N_QUBITS, n_params=25)
        
        # Classical layers
        self.fc1 = nn.Linear(1, 512)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(512, 2)  # Output: [non-face (0), face (1)]
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, 10) PCA features
        
        Returns:
            (batch, 2) logits
        """
        # Quantum layer
        x = self.qlayer(x)  # (batch, 1)
        
        # Classical layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
    
    def extract_features(self, x):
        """
        Extract 512-dim embeddings for recognition
        
        Args:
            x: (batch, 10) PCA features
        
        Returns:
            (batch, 512) feature embeddings
        """
        x = self.qlayer(x)  # (batch, 1)
        x = self.fc1(x)     # (batch, 512)
        return x
