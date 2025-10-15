"""
Simple neural network models using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    A simple feedforward neural network for demonstration purposes.
    
    Architecture:
        - Input layer
        - Hidden layer with ReLU activation
        - Output layer
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden layer
        output_size (int): Size of output layer
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    """
    A simple convolutional neural network for image classification.
    
    Architecture:
        - Conv layer 1: 1 -> 32 channels
        - Conv layer 2: 32 -> 64 channels
        - Fully connected layers
    
    Args:
        num_classes (int): Number of output classes
    """
    
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
