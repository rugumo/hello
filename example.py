"""
Example script demonstrating basic usage of the PyTorch models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from hello import SimpleNet, ConvNet
from hello.utils import train_step, evaluate


def example_simple_net():
    """Example using SimpleNet with random data"""
    print("=" * 50)
    print("SimpleNet Example")
    print("=" * 50)
    
    # Create model
    model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Generate random data
    batch_size = 32
    data = torch.randn(batch_size, 784)
    target = torch.randint(0, 10, (batch_size,))
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining...")
    for epoch in range(5):
        loss = train_step(model, data, target, optimizer, criterion)
        print(f"Epoch {epoch + 1}/5, Loss: {loss:.4f}")
    
    # Evaluation
    eval_loss, accuracy = evaluate(model, data, target, criterion)
    print(f"\nEvaluation - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 784)
        output = model(test_input)
        prediction = output.argmax(dim=1)
        print(f"\nSample prediction: {prediction.item()}")


def example_conv_net():
    """Example using ConvNet with random image data"""
    print("\n" + "=" * 50)
    print("ConvNet Example")
    print("=" * 50)
    
    # Create model
    model = ConvNet(num_classes=10)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Generate random image data (28x28 grayscale images)
    batch_size = 16
    data = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 10, (batch_size,))
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining...")
    for epoch in range(5):
        loss = train_step(model, data, target, optimizer, criterion)
        print(f"Epoch {epoch + 1}/5, Loss: {loss:.4f}")
    
    # Evaluation
    eval_loss, accuracy = evaluate(model, data, target, criterion)
    print(f"\nEvaluation - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        prediction = output.argmax(dim=1)
        print(f"\nSample prediction: {prediction.item()}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    
    # Run examples
    example_simple_net()
    example_conv_net()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)
