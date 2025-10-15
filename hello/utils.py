"""
Utility functions for training and evaluation
"""

import torch
import torch.nn as nn
from typing import Tuple


def train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Perform a single training step.
    
    Args:
        model: PyTorch model
        data: Input data tensor
        target: Target labels tensor
        optimizer: Optimizer
        criterion: Loss function
    
    Returns:
        float: Loss value for this step
    """
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate the model on given data.
    
    Args:
        model: PyTorch model
        data: Input data tensor
        target: Target labels tensor
        criterion: Loss function
    
    Returns:
        Tuple[float, float]: Loss and accuracy
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
    return loss, accuracy
