import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# File paths
MODEL_PATH = "/workspaces/Hybrid-Quantum-Classical-AI/lstm_model_final.pth"
DATA_PATH = "/workspaces/Hybrid-Quantum-Classical-AI/datasets/processed/train.csv"

# Define Dataset Class
class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        if "text" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")
        self.texts = self.data["text"].values
        self.labels = self.data["label"].values
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# Define Model Class (Matching LSTM Architecture)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return self.softmax(x)

# Load Model
def load_model():
    try:
        # Attempt to load as state_dict
        model = LSTMModel(vocab_size=10000, embedding_dim=128, hidden_dim=256, output_dim=2)
        model.load_state_dict(torch.load(MODEL_PATH))
    except TypeError:
        # If the saved file is the entire model, load it directly
        model = torch.load(MODEL_PATH)
    model.eval()
    return model

# Run Inference
def predict(model, text):
    # Dummy Tokenization - Replace with real tokenizer if needed
    tokens = torch.tensor([ord(char) % 100 for char in text]).unsqueeze(0)  # Convert text to numeric tensor
    output = model(tokens)
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# Main Execution
if __name__ == "__main__":
    # Load model
    lstm_model = load_model()
    print("Model Loaded Successfully!")

    # Load dataset
    dataset = TextDataset(DATA_PATH)
    print(f"Number of samples: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example: Running inference on a sample
    sample_text = dataset.texts[0]
    predicted = predict(lstm_model, sample_text)
    print(f"Predicted label for: '{sample_text}' -> {predicted}")
    
    # Save the model
    torch.save(lstm_model.state_dict(), MODEL_PATH)
    print(f"MODEL SAVED TO {MODEL_PATH}")
