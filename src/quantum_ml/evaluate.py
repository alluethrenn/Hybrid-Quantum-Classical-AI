import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from quantum_layers import HybridModel
from torch.utils.tensorboard import SummaryWriter
import logging.config
import yaml
import os

# Load logging config
with open("configs/logging.yaml", "r") as file:
    logging_config = yaml.safe_load(file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Function to load MNIST
def load_mnist_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Download the dataset
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Load config
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Evaluate function
def evaluate_model(config):
    # Load test data
    _, test_loader = load_mnist_data(config["data"]["batch_size"])

    # Initialize the model
    model = HybridModel(
        num_qubits=config["quantum"]["num_qubits"],
        num_layers=config["quantum"]["layers"],
        hidden_layers=config["classical_nn"]["hidden_layers"],
        output_size=10  # 10 output classes for digits 0-9
    )
    model.load_state_dict(torch.load(config["training"]["checkpoint_dir"] + "/hybrid_model.pth"))
    model.eval()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")

# Main function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Quantum-Classical Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config and evaluate
    config = load_config(args.config)
    evaluate_model(config)
