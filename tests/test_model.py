"""Tests for PyTorch models"""

import torch
import pytest
from hello.model import SimpleNet, ConvNet


class TestSimpleNet:
    """Test cases for SimpleNet"""
    
    def test_model_creation(self):
        """Test that model can be created with default parameters"""
        model = SimpleNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_with_custom_parameters(self):
        """Test model creation with custom parameters"""
        model = SimpleNet(input_size=100, hidden_size=50, output_size=5)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass with random input"""
        model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
        batch_size = 4
        x = torch.randn(batch_size, 784)
        output = model(x)
        assert output.shape == (batch_size, 10)
    
    def test_output_range(self):
        """Test that output has correct dimensions"""
        model = SimpleNet()
        x = torch.randn(1, 784)
        output = model(x)
        assert output.shape[1] == 10
    
    def test_backward_pass(self):
        """Test that backward pass works"""
        model = SimpleNet()
        x = torch.randn(4, 784)
        target = torch.randint(0, 10, (4,))
        criterion = torch.nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None


class TestConvNet:
    """Test cases for ConvNet"""
    
    def test_model_creation(self):
        """Test that model can be created"""
        model = ConvNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_with_custom_classes(self):
        """Test model creation with custom number of classes"""
        model = ConvNet(num_classes=5)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass with image input"""
        model = ConvNet(num_classes=10)
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10)
    
    def test_output_shape(self):
        """Test output has correct shape"""
        model = ConvNet(num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_dropout_training_mode(self):
        """Test that dropout works in training mode"""
        model = ConvNet()
        model.train()
        x = torch.randn(2, 1, 28, 28)
        output1 = model(x)
        output2 = model(x)
        # Outputs should be different due to dropout
        # (may rarely fail due to randomness, but very unlikely)
        assert not torch.allclose(output1, output2, atol=1e-6)
    
    def test_dropout_eval_mode(self):
        """Test that dropout is disabled in eval mode"""
        model = ConvNet()
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)
