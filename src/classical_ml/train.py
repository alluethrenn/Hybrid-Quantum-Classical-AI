# src/classical_ml/train.py

import os
import pickle
import numpy as np
from datasets import load_mnist  # Or any dataset from datasets.py
from models import get_logistic_regression, get_svm, get_random_forest
from sklearn.metrics import accuracy_score

MODEL_PATH = "results/classical_ml_model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Create results directory

def save_model(model, model_path):
    """Save the trained model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def train_model(model_type='logistic_regression', batch_size=32):
    # Load data
    train_loader = load_mnist(batch_size=batch_size)
    
    # Choose the model
    if model_type == 'logistic_regression':
        model = get_logistic_regression()
    elif model_type == 'svm':
        model = get_svm()
    elif model_type == 'random_forest':
        model = get_random_forest()
    else:
        raise ValueError("Unknown model type.")
    
    # Convert data to numpy for sklearn models
    train_data, train_labels = [], []
    for data, labels in train_loader:
        train_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        train_labels.append(labels.numpy())

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Train the model
    model.fit(train_data, train_labels)

    # Save the trained model
    save_model(model, MODEL_PATH)
    return model

def load_model(model_path):
    """Load a saved model."""
    if not os.path.exists(model_path):
        print("No model file found, training a new model...")
        return train_model()  # Train the model if no saved model exists
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model

if __name__ == "__main__":
    load_model(MODEL_PATH)
