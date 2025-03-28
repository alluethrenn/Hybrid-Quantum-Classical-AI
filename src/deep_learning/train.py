# src/deep_learning/train.py

import torch
import torch.optim as optim
import torch.nn as nn
import os
from src.deep_learning.models.model import DeepLearningModel  # Updated import path
from datasets import get_mnist_dataloader

CHECKPOINT_DIR = "results/checkpoints"
MODEL_PATH = "results/deep_learning_model.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure checkpoint directory exists

def save_checkpoint(model, epoch, loss, optimizer):
    """Save model checkpoint."""
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def train_model(epochs=10, lr=0.001, batch_size=32):
    train_loader, _ = get_mnist_dataloader(batch_size)
    
    model = DeepLearningModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')  # Track the best loss

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Save checkpoint every epoch
        save_checkpoint(model, epoch+1, epoch_loss, optimizer)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved!")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
