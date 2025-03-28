# src/classical_ml/evaluate.py

import os  # Add this line
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

FEATURE_TRANSFORMS_PATH = "src/classical_ml/models/feature_transformations.joblib"

def load_model(model_type):
    """Load a pre-trained model based on the model type."""
    model_path = MODEL_PATHS.get(model_type)
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        return joblib.load(model_path)
    else:
        print(f"No pre-trained model found for {model_type}. Please train the model first.")
        return None

def load_feature_transformations():
    """Load feature transformations (e.g., scaler)."""
    if os.path.exists(FEATURE_TRANSFORMS_PATH):
        with open(FEATURE_TRANSFORMS_PATH, 'rb') as f:
            return joblib.load(f)
    else:
        print("No feature transformations found. Please train the model first.")
        return None

def save_feature_transformations(scaler, path):
    """Save feature transformations to a file."""
    with open(path, 'wb') as f:
        joblib.dump(scaler, f)

def save_model(model, path):
    """Save the trained model to a file."""
    joblib.dump(model, path)

def get_logistic_regression():
    """Return a logistic regression model."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression()

def get_svm():
    """Return an SVM model."""
    from sklearn.svm import SVC
    return SVC()

def get_random_forest():
    """Return a random forest model."""
    from sklearn.ensemble import RandomForestClassifier
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

def evaluate_model(model_type='logistic_regression'):
    """Evaluate the model and log results."""
    model = load_model(model_type)
    if not model:
        print("Error: Model is not loaded. Cannot proceed with evaluation.")
        return

    # Load the feature transformations
    scaler = load_feature_transformations()
    if not scaler:
        print("Error: Feature transformations not loaded. Cannot proceed.")
        return

    # Load test data
    test_loader = load_mnist(train=False)
    test_data, test_labels = [], []
    for data, labels in test_loader:
        test_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        test_labels.append(labels.numpy())

    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)

    # Apply feature transformations to the test data
    test_data = scaler.transform(test_data)

    # Get predictions and evaluate accuracy
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model('logistic_regression')  # Change this to test other models

