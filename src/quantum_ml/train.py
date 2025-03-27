import torch
import torch.nn as nn
import torch.optim as optim
import logging.config
import yaml
import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from quantum_layers import QuantumLayer, DenseQuantumLayer, QuantumActivationLayer  # Quantum layers from earlier
import pennylane as qml

# Load logging config
with open("configs/logging.yaml", "r") as file:
    logging_config = yaml.safe_load(file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Define Hybrid Quantum-Classical Model
class HybridModel(nn.Module):
    def __init__(self, num_qubits, num_layers, hidden_layers, output_size):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Define classical layers
        self.classical_layers = nn.ModuleList()
        for in_features, out_features in zip([self.num_qubits] + self.hidden_layers[:-1], self.hidden_layers):
            self.classical_layers.append(nn.Linear(in_features, out_features))
        self.classical_layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        # Define quantum layers
        self.quantum_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.quantum_layers.append(DenseQuantumLayer(self.num_qubits, self.num_qubits))

    def forward(self, x):
        # Classical layers
        for layer in self.classical_layers:
            x = torch.relu(layer(x))

        # Quantum layers (using PennyLane)
        for quantum_layer in self.quantum_layers:
            x = quantum_layer.forward(x)

        return x

# Load config
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Training function
def train_model(config):
    # Load dataset (use DataLoader here)
    from evaluate import load_mnist_data
    train_loader = load_mnist_data(config['data']['batch_size'])

    # Initialize model, optimizer, and loss function
    model = HybridModel(
        num_qubits=config["quantum"]["num_qubits"],
        num_layers=config["quantum"]["layers"],
        hidden_layers=config["classical_nn"]["hidden_layers"],
        output_size=1
    )
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.MSELoss()

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config["training"]["log_dir"])

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_fn(outputs, labels.unsqueeze(1))  # Ensure the output is in the correct shape

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {avg_loss:.4f}")

        # Log loss to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)

    logger.info("Training completed!")
    # Save the trained model
    torch.save(model.state_dict(), config["training"]["checkpoint_dir"] + "/hybrid_model.pth")
    writer.close()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Quantum-Classical Model")
    parser.add_argument("--config", type=str, required=True, help=("configs/config.yaml")
    args = parser.parse_args()

    # Load config and train
    config = load_config(args.config)
    train_model(config)
