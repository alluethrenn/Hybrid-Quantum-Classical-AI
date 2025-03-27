# filepath: src/quantum_ml/quantum_layers.py

import pennylane as qml
import torch
import torch.nn as nn

# Define a basic Quantum Layer using PennyLane
class QuantumLayer(nn.Module):
    def __init__(self, num_qubits):
        super(QuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_qubits)

    def forward(self, inputs):
        # Convert the input tensor to a suitable format for quantum circuits
        inputs = inputs.tolist()  # Convert tensor to list for easier manipulation
        
        @qml.qnode(self.dev)
        def quantum_circuit():
            # Apply basic quantum operations (e.g., Hadamard, CNOT, etc.)
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # Execute the quantum circuit
        quantum_output = quantum_circuit()
        return torch.tensor(quantum_output)

# Dense Quantum Layer: Combines quantum and classical weights
class DenseQuantumLayer(QuantumLayer):
    def __init__(self, num_qubits, num_outputs):
        super().__init__(num_qubits)
        self.num_outputs = num_outputs
        self.weights = torch.nn.Parameter(torch.zeros(num_qubits, num_outputs))

    def forward(self, inputs):
        quantum_outputs = super().forward(inputs)
        
        # Classic linear transformation based on quantum output
        outputs = torch.matmul(quantum_outputs, self.weights)
        return outputs

# Quantum Activation Layer (e.g., Sigmoid, ReLU)
class QuantumActivationLayer(QuantumLayer):
    def __init__(self, num_qubits, activation_function):
        super().__init__(num_qubits)
        self.activation_function = activation_function

    def forward(self, inputs):
        quantum_outputs = super().forward(inputs)
        
        if self.activation_function == "sigmoid":
            return torch.sigmoid(quantum_outputs)
        elif self.activation_function == "relu":
            return torch.relu(quantum_outputs)
        elif self.activation_function == "tanh":
            return torch.tanh(quantum_outputs)
        else:
            raise ValueError(f"Activation function {self.activation_function} not recognized.")

