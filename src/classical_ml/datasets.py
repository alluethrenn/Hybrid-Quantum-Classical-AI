# src/classical_ml/datasets.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist(batch_size=32, train=True, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_cifar10(batch_size=32, train=True, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Additional dataset functions can be added here as needed.
