"""Tests for utility functions"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from hello.model import SimpleNet
from hello.utils import train_step, evaluate


class TestTrainStep:
    """Test cases for train_step function"""
    
    def test_train_step_returns_loss(self):
        """Test that train_step returns a loss value"""
        model = SimpleNet()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        data = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        
        loss = train_step(model, data, target, optimizer, criterion)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_train_step_updates_weights(self):
        """Test that train_step updates model weights"""
        model = SimpleNet()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Get initial weights
        initial_weights = [p.clone() for p in model.parameters()]
        
        data = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        
        train_step(model, data, target, optimizer, criterion)
        
        # Check that weights have changed
        for initial, current in zip(initial_weights, model.parameters()):
            assert not torch.allclose(initial, current)


class TestEvaluate:
    """Test cases for evaluate function"""
    
    def test_evaluate_returns_loss_and_accuracy(self):
        """Test that evaluate returns loss and accuracy"""
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        
        data = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        
        loss, accuracy = evaluate(model, data, target, criterion)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss >= 0
        assert 0 <= accuracy <= 1
    
    def test_evaluate_no_gradient(self):
        """Test that evaluate doesn't compute gradients"""
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        
        data = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        
        evaluate(model, data, target, criterion)
        
        # Gradients should be None since we're in eval mode
        for param in model.parameters():
            assert param.grad is None
    
    def test_evaluate_sets_eval_mode(self):
        """Test that evaluate sets model to eval mode"""
        model = SimpleNet()
        model.train()  # Set to train mode first
        criterion = nn.CrossEntropyLoss()
        
        data = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        
        evaluate(model, data, target, criterion)
        
        # Model should be in eval mode after evaluation
        assert not model.training
