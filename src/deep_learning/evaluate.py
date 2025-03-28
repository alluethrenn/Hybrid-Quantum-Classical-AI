# src/deep_learning/evaluate.py

import torch
import torch.nn as nn
from model import DeepLearningModel
from datasets import get_mnist_dataloader

def evaluate_model(model_path="results/deep_learning_model.pth"):
    _, test_loader = get_mnist_dataloader()

    model = DeepLearningModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}% - Loss: {test_loss/len(test_loader)}")

if __name__ == "__main__":
    evaluate_model()
