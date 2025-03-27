import torch
import torch.nn as nn
import logging.config
import yaml
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from train import HybridModel  # Import the model from train.py
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Load logging config
with open("configs/logging.yaml", "r") as file:
    logging_config = yaml.safe_load(file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# 1. Load DataLoader for MNIST or Custom Dataset
def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download and load the test data (MNIST dataset)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Custom dataset loader (if you have a CSV file)
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Read CSV file
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming CSV has 'features' and 'labels' columns
        sample = self.data.iloc[idx, :-1].values  # Features (exclude label column)
        label = self.data.iloc[idx, -1]           # Label column
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def load_custom_data(csv_file, batch_size):
    dataset = CustomDataset(csv_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

# 2. Evaluate model
def evaluate_model(config):
    logger.info("Loading test dataset...")
    
    # Load dataset based on config (either MNIST or custom CSV dataset)
    if "test_data_path" in config["data"]:
        test_loader = load_custom_data(config["data"]["test_data_path"], config["data"]["batch_size"])
    else:
        test_loader = load_mnist_data(config["data"]["batch_size"])

    if not test_loader:
        logger.error("Failed to load test dataset.")
        return

    # Initialize model
    model = HybridModel(
        num_qubits=config["quantum"]["num_qubits"],
        num_layers=config["quantum"]["layers"],
        hidden_layers=config["classical_nn"]["hidden_layers"],
        output_size=1
    )

    # Load trained model
    model_path = os.path.join(config["training"]["checkpoint_dir"], "hybrid_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config["training"]["log_dir"])

    # Evaluation loop
    logger.info("Starting evaluation...")
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Assuming binary classification
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/evaluation", nn.MSELoss()(torch.tensor(y_pred), torch.tensor(y_true)).item(), 0)
    writer.add_scalar("Accuracy/evaluation", accuracy, 0)
    writer.add_scalar("Precision/evaluation", precision, 0)
    writer.add_scalar("Recall/evaluation", recall, 0)

    logger.info(f"Evaluation Loss: {nn.MSELoss()(torch.tensor(y_pred), torch.tensor(y_true)).item():.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")

    # Close TensorBoard writer
    writer.close()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Quantum-Classical Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config and evaluate
    config = load_config(args.config)
    evaluate_model(config)



