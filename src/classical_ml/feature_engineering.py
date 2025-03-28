# src/classical_ml/feature_engineering.py

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from datasets import load_mnist  # Replace with your dataset loading function

FEATURE_TRANSFORMS_PATH = "src/classical_ml/models/feature_transformations.joblib"

def load_and_process_data(batch_size=32):
    """Load dataset and apply feature engineering (scaling, PCA, one-hot encoding)."""
    # Load data
    train_loader = load_mnist(batch_size=batch_size)

    # Flatten images and prepare data
    train_data, train_labels = [], []
    for data, labels in train_loader:
        train_data.append(data.view(data.size(0), -1).numpy())  # Flatten images
        train_labels.append(labels.numpy())

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Apply feature engineering steps
    processed_data = apply_feature_engineering(train_data, train_labels)

    # Save the feature transformations (scalers, PCA, etc.)
    save_feature_transformations()

    return processed_data, train_labels

def apply_feature_engineering(data, train_labels):
    """Apply various feature engineering steps like scaling, PCA, and polynomial features."""
    
    # Step 1: Handle missing values (if any)
    imputer = SimpleImputer(strategy='mean')  # Replace missing values with mean
    data = imputer.fit_transform(data)

    # Step 2: Scale the data (standardize to have mean=0, variance=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Step 3: Apply PCA for dimensionality reduction (optional based on your data)
    pca = PCA(n_components=50)  # Reduce to 50 components
    data = pca.fit_transform(data)

    # Step 4: Generate polynomial features (if needed for higher-order interactions)
    poly = PolynomialFeatures(degree=2)  # Quadratic features
    data = poly.fit_transform(data)

    # Step 5: Remove outliers using Z-score (or IQR) method
    data = remove_outliers(data)

    # Step 6: Apply feature selection using Recursive Feature Elimination (RFE)
    model = LogisticRegression(max_iter=1000)  # Simple logistic regression for RFE
    selector = RFE(model, n_features_to_select=50, step=1)
    data = selector.fit_transform(data, train_labels)

    # Step 7: Optional - Apply log transformations to skewed data
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    data = log_transformer.fit_transform(data)

    # Step 8: One-hot encode categorical features (if any)
    # Example: One-hot encode categorical features (if present)
    encoder = OneHotEncoder(sparse=False)
    categorical_data = encoder.fit_transform(data)

    # Store transformers for later use
    store_transformers(scaler, pca, poly)

    return data

def remove_outliers(data, threshold=3):
    """Remove outliers using Z-score."""
    from scipy.stats import zscore
    z_scores = zscore(data)
    filtered_data = data[(np.abs(z_scores) < threshold).all(axis=1)]
    
    # Check if the filtered data is empty
    if filtered_data.shape[0] == 0:
        print("Warning: All data points were removed as outliers. Skipping outlier removal.")
        return data  # Return the original data if all points are removed
    
    return filtered_data

def save_feature_transformations():
    """Save all feature transformations used in the data preprocessing."""
    # Save scaler, PCA, and polynomial features
    with open(FEATURE_TRANSFORMS_PATH, 'wb') as f:
        joblib.dump({
            'scaler': StandardScaler(),
            'pca': PCA(n_components=50),
            'poly': PolynomialFeatures(degree=2)
        }, f)
    print(f"Feature transformations saved to {FEATURE_TRANSFORMS_PATH}")

def store_transformers(scaler, pca, poly):
    """Store transformers (scaler, PCA, etc.) to be used later."""
    with open(FEATURE_TRANSFORMS_PATH, 'wb') as f:
        joblib.dump({'scaler': scaler, 'pca': pca, 'poly': poly}, f)
    print("Transformers (scaler, PCA, polynomial features) saved.")

def load_feature_transformations():
    """Load previously saved feature transformations (scaler, PCA, etc.)."""
    if os.path.exists(FEATURE_TRANSFORMS_PATH):
        with open(FEATURE_TRANSFORMS_PATH, 'rb') as f:
            return joblib.load(f)
    else:
        print("No feature transformations found. Please train the model first.")
        return None

if __name__ == "__main__":
    # Process data and save transformations
    train_data, train_labels = load_and_process_data()
    print("Data processed and saved feature transformations.")
