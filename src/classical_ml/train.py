# src/classical_ml/train.py

import os
import joblib
import numpy as np
from datasets import load_mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_PATHS = {
    "decision_tree": "src/classical_ml/models/best_decision_tree_model.joblib",
    "logistic_regression": "src/classical_ml/models/best_logistic_regression_model.joblib",
    "mlp": "src/classical_ml/models/best_mlp_model.joblib",
    "random_forest": "src/classical_ml/models/best_random_forest_model.joblib",
    "svm": "src/classical_ml/models/best_svm_model.joblib",
}

FEATURE_TRANSFORMS_PATH = "src/classical_ml/models/feature_transformations.joblib"  # Save transformation here

def save_model(model, model_path):
    """Save the trained model."""
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)
    print(f"Model saved to {model_path}")

def save_feature_transformations(scaler, path):
    """Save feature transformations (like scaling)"""
    with open(path, 'wb') as f:
        joblib.dump(scaler, f)
    print(f"Feature transformations saved to {path}")

def load_feature_transformations(path):
    """Load feature transformations"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return joblib.load(f)
    else:
        print(f"No feature transformations found at {path}")
        return None

def train_model(model_type='logistic_regression', batch_size=32):
    # Load data
    train_loader = load_mnist(batch_size=batch_size)

    # Flatten the images and prepare training data
    train_data, train_labels = [], []
    for data, labels in train_loader:
        train_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        train_labels.append(labels.numpy())

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Feature transformation (e.g., scaling)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    # Save the transformations
    save_feature_transformations(scaler, FEATURE_TRANSFORMS_PATH)

    # Choose the model
    if model_type == 'logistic_regression':
        model = get_logistic_regression()
    elif model_type == 'svm':
        model = get_svm()
    elif model_type == 'random_forest':
        model = get_random_forest()
    else:
        raise ValueError("Unknown model type.")
    
    # Train the model
    model.fit(train_data, train_labels)

    # Save the trained model
    save_model(model, MODEL_PATHS[model_type])

    return model

if __name__ == "__main__":
    train_model('logistic_regression')  # Or change this to train other models
