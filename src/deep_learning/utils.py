# src/deep_learning/utils.py

import torch
from model import DeepLearningModel

def load_trained_model(model_path="results/deep_learning_model.pth"):
    """Loads a trained deep learning model."""
    model = DeepLearningModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
