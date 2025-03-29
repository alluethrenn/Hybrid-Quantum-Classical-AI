
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Clean the data
def clean_data(df):
    df = df.drop_duplicates()  # Remove duplicate rows
    df = df.dropna()  # Drop rows with missing values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Trim whitespace
    return df

# Split data into train and test sets
def split_data(df, test_size=0.5, random_state=42,):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

# Save the split data
def save_data(train, test, train_path, test_path):
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

if __name__ == "__main__":
    input_file = "/workspaces/Hybrid-Quantum-Classical-AI/datasets/raw/Sheet_1.csv"  # Change to your file path
    train_file = "train.csv"
    test_file = "test.csv"

    df = load_csv(input_file)
    df_clean = clean_data(df)
    train, test = split_data(df_clean)

    save_data(train, test, train_file, test_file)
    print("Data processing complete. Train and test files saved.")
