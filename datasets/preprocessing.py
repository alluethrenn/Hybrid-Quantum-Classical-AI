import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# Paths to save preprocessing models
PREPROCESSOR_DIR = "preprocessors"
os.makedirs(PREPROCESSOR_DIR, exist_ok=True)

def load_csv(file_paths):
    """Load multiple CSV files into a single Pandas DataFrame."""
    if isinstance(file_paths, str):
        # If a single file path is provided, load it directly
        return pd.read_csv(file_paths, encoding="ISO-8859-1")  # Specify encoding
    elif isinstance(file_paths, (list, tuple)):
        # If multiple file paths are provided, load and concatenate them
        dataframes = [pd.read_csv(file_path, encoding="ISO-8859-1") for file_path in file_paths]
        return pd.concat(dataframes, ignore_index=True)
    else:
        raise ValueError("Invalid file path or buffer object type.")

def clean_data(df):
    """Remove duplicates and strip whitespace from string columns."""
    df = df.drop_duplicates()
    df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)
    return df

def preprocess_data(df, save_models=True):
    """Preprocess data by imputing, encoding categorical variables, and scaling numeric features."""
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')

    # Remove irrelevant columns (e.g., Unnamed columns)
    cat_cols = [col for col in cat_cols if not col.startswith("Unnamed")]

    # Handle missing values
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Impute only columns with at least one non-missing value
    num_cols = [col for col in num_cols if col in df.columns and df[col].notna().any()]
    cat_cols = [col for col in cat_cols if col in df.columns and df[col].notna().any()]

    if num_cols:
        try:
            df.loc[:, num_cols] = num_imputer.fit_transform(df[num_cols])
            print(f"Imputed missing values in numerical columns: {num_cols}")
        except Exception as e:
            print(f"Error imputing numerical columns: {e}")
    else:
        print("No numerical columns to impute.")

    if cat_cols:
        try:
            df.loc[:, cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            print(f"Imputed missing values in categorical columns: {cat_cols}")
        except Exception as e:
            print(f"Error imputing categorical columns: {e}")
    else:
        print("No categorical columns to impute.")

    # Save imputers
    if save_models:
        joblib.dump(num_imputer, os.path.join(PREPROCESSOR_DIR, "num_imputer.pkl"))
        joblib.dump(cat_imputer, os.path.join(PREPROCESSOR_DIR, "cat_imputer.pkl"))

    # Encode categorical variables
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Updated parameter
        encoded_array = encoder.fit_transform(df[cat_cols])
        print(f"Encoded array shape: {encoded_array.shape}")
        print(f"Original DataFrame index: {df.index.shape}")
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

        # Save encoder
        if save_models:
            joblib.dump(encoder, os.path.join(PREPROCESSOR_DIR, "encoder.pkl"))
    else:
        encoded_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no categorical columns

    # Standardize numerical features
    if num_cols:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[num_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=num_cols, index=df.index)

        # Save scaler
        if save_models:
            joblib.dump(scaler, os.path.join(PREPROCESSOR_DIR, "scaler.pkl"))
    else:
        scaled_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no numerical columns

    # Combine processed numerical and categorical data
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)

    # Save column names for future inference
    if save_models:
        joblib.dump(processed_df.columns.tolist(), os.path.join(PREPROCESSOR_DIR, "columns.pkl"))

    return processed_df

def split_data(df, test_size=0.2, random_state=42):
    """Split the data into training and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)

def save_data(train, test, train_path, test_path):
    """Save train and test datasets to CSV files."""
    print(f"Saving train data to {train_path}")
    print(f"Saving test data to {test_path}")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

if __name__ == "__main__":
    # Define input and output file paths
    input_files = [
        "/workspaces/Hybrid-Quantum-Classical-AI/code_dataset.csv"
    ]  # List of dataset paths
    output_dir = "/workspaces/Hybrid-Quantum-Classical-AI/datasets/processed"  # Desired output directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")

    # Load, clean, preprocess, and split the data
    df = load_csv(input_files)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    train, test = split_data(df_processed)

    # Save the train and test datasets
    save_data(train, test, train_file, test_file)
    print("Data processing complete. Train and test files saved.")
    print(f"Preprocessing models saved in '{PREPROCESSOR_DIR}'")
