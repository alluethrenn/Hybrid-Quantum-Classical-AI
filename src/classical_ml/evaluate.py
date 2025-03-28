# src/classical_ml/evaluate.py

import joblib
import numpy as np
from datasets import load_mnist
from sklearn.metrics import accuracy_score

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

