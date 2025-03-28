# src/deep_learning/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLearningModel(nn.Module):
    def __init__(self, input_size=28*28, hidden_layers=[128, 64], output_size=10):
        super(DeepLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
