import torch
import torch.nn as nn
import logging.config
import yaml
import argparse
import os
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
    logger.info("Loading dataset...")
    # Dummy dataset for evaluation (use test set here)
    X_test = torch.rand((config_

