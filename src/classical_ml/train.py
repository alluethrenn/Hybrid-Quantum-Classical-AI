# src/classical_ml/train.py

import os
import joblib
import numpy as np
from datasets import load_mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

def get_logistic_regression():
    """Return a Logistic Regression model."""
    return LogisticRegression(max_iter=1000)

def get_svm():
    """Return an SVM model."""
    return SVC()

def get_random_forest():
    """Return a Random Forest model."""
    return RandomForestClassifier()

def train_model(model_type='logistic_regression', batch_size=32):
    # Load data
    mnist_data = load_mnist()
    
    # Extract data and labels from the DataLoader
    train_data, train_labels = [], []
    for batch in mnist_data:
        data, labels = batch
        train_data.append(data.numpy())  # Convert tensors to NumPy arrays
        train_labels.append(labels.numpy())
    
    # Concatenate all batches into a single array
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Flatten the images and prepare training data
    train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten images

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
