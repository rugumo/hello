# Hello - Pure PyTorch Project

A pure PyTorch project demonstrating fundamental deep learning concepts with clean, modular code.

## Features

- **Pure PyTorch Implementation**: Built entirely with PyTorch for maximum flexibility
- **Modular Architecture**: Clean separation of models, utilities, and examples
- **Multiple Model Types**: Includes both feedforward and convolutional neural networks
- **Comprehensive Tests**: Full test coverage with pytest
- **Example Scripts**: Ready-to-run examples demonstrating model usage

## Project Structure

```
hello/
├── hello/              # Main package
│   ├── __init__.py    # Package initialization
│   ├── model.py       # Neural network models
│   └── utils.py       # Training and evaluation utilities
├── tests/             # Test suite
│   ├── test_model.py  # Model tests
│   └── test_utils.py  # Utility function tests
├── example.py         # Example usage script
├── requirements.txt   # Project dependencies
├── pyproject.toml    # Project configuration
└── README.md         # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rugumo/hello.git
cd hello
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

4. (Optional) Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Running the Example

The project includes a comprehensive example script that demonstrates both models:

```bash
python example.py
```

This will:
- Create and train a SimpleNet (feedforward network)
- Create and train a ConvNet (convolutional network)
- Show training progress and evaluation results
- Demonstrate inference with sample predictions

### Using in Your Code

```python
import torch
from hello import SimpleNet, ConvNet
from hello.utils import train_step, evaluate

# Create a simple feedforward network
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)

# Or create a convolutional network
conv_model = ConvNet(num_classes=10)

# Prepare your data
data = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))

# Setup optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train
loss = train_step(model, data, target, optimizer, criterion)

# Evaluate
eval_loss, accuracy = evaluate(model, data, target, criterion)
```

## Models

### SimpleNet

A basic feedforward neural network with:
- Configurable input, hidden, and output sizes
- Single hidden layer with ReLU activation
- Suitable for simple classification tasks

**Parameters:**
- `input_size` (int): Size of input features (default: 784)
- `hidden_size` (int): Size of hidden layer (default: 128)
- `output_size` (int): Size of output layer (default: 10)

### ConvNet

A convolutional neural network for image classification with:
- Two convolutional layers with max pooling
- Dropout for regularization
- Fully connected layers for classification

**Parameters:**
- `num_classes` (int): Number of output classes (default: 10)

**Expected Input:** Images of shape (batch_size, 1, 28, 28)

## Testing

Run the test suite with pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=hello --cov-report=term-missing

# Run specific test file
pytest tests/test_model.py
```

## Development

### Code Style

This project follows standard Python conventions:
- Type hints for better code clarity
- Docstrings for all public functions and classes
- Modular design for easy extension

### Adding New Models

1. Add your model class to `hello/model.py`
2. Export it in `hello/__init__.py`
3. Create tests in `tests/test_model.py`
4. Update the README with usage examples

## Requirements

- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - PyTorch vision library
- `numpy>=1.24.0` - Numerical computing library

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with [PyTorch](https://pytorch.org/) - An open source machine learning framework.