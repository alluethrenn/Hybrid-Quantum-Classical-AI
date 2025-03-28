# src/deep_learning/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import DeepLearningModel
from datasets import get_mnist_dataloader

def train_model(epochs=10, lr=0.001, batch_size=32):
    train_loader, _ = get_mnist_dataloader(batch_size)
    
    model = DeepLearningModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), "results/deep_learning_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model()

