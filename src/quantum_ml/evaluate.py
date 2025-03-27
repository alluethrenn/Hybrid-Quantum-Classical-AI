import torch
import torch.nn as nn
import logging.config
import yaml
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from train import HybridModel  # Import the model from train.py

# Load logging config
with open("configs/logging.yaml", "r") as file:
    logging_config = yaml.safe_load(file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Evaluate model
def evaluate_model(config):
    logger.info("Loading test dataset...")
    # Dummy dataset for evaluation (use actual test data here)
    # For example, replace this with a DataLoader
    X_test = torch.rand((config["data"]["batch_size"], config["quantum"]["num_qubits"]))
    y_test = torch.rand((config["data"]["batch_size"], 1))

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
    y_pred = []
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()  # Assuming binary classification
        y_pred.extend(predicted.cpu().numpy())
        y_true = y_test.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/evaluation", nn.MSELoss()(outputs, y_test).item(), 0)
    writer.add_scalar("Accuracy/evaluation", accuracy, 0)
    writer.add_scalar("Precision/evaluation", precision, 0)
    writer.add_scalar("Recall/evaluation", recall, 0)

    logger.info(f"Evaluation Loss: {nn.MSELoss()(outputs, y_test).item():.4f}")
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


