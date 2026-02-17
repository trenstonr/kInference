# C++ Inference Engine

A lightweight neural network inference engine written in C++ for deployment on embedded systems, particularly the NVIDIA Jetson TX2.

## Overview

Custom inference engine for deploying PyTorch models in C++. Supports arbitrary multilayer fully connected networks with automatic activation configuration.

## Architecture

### Tensor Class (`include/tensor.hpp`, `src/tensor.cpp`)
- N-dimensional array container
- Binary file loading for PyTorch-exported weights
- Memory management and reshape operations

### Engine Class (`include/engine.hpp`, `src/engine.cpp`)
- Matrix operations: matmul, element-wise add/multiply
- Activation functions: ReLU, Sigmoid, Softmax

### Inferencer Class (`include/inferencer.hpp`, `src/inferencer.cpp`)
- Supports arbitrary number of fully connected layers
- Automatic activation configuration: ReLU for hidden layers, Softmax for output
- Loads weights and biases from binary files
- Executes forward pass through all layers

## Project Structure

```
.
├── include/
│   ├── tensor.hpp          # Tensor class definition
│   ├── engine.hpp          # Matrix and tensor operations
│   └── inferencer.hpp      # Inference pipeline
├── src/
│   ├── tensor.cpp
│   ├── engine.cpp
│   └── inferencer.cpp
├── python/
│   ├── train_mnist.py      # PyTorch script: 2-layer network on MNIST (784→128→10)
│   ├── train_cifar10.py    # PyTorch script: 3-layer network on CIFAR-10 (3072→256→128→10)
└── main.cpp                # Example 3-layer CIFAR-10 inference
```

## Usage

### Example: CIFAR-10 with 3-layer Network (Default)

Train and export the 3-layer CIFAR-10 model:

```bash
cd python
python3 train_cifar10.py
cd ..
```

This creates `exported_data_cifar10/` with 3 sets of weights/biases, test images, and expected outputs.

Compile and run the C++ inference:

```bash
g++ -std=c++17 main.cpp src/* -I./include -o k_infer
./k_infer
```

Example output:

```
Test Image 0:
  Prediction: 0.0225 0.1576 0.0984 0.3352 0.0338 0.1224 0.0932 0.0374 0.0595 0.0400 
  Expected:   0.0225 0.1576 0.0984 0.3352 0.0338 0.1224 0.0932 0.0374 0.0595 0.0400
```

## Roadmap

### CUDA Acceleration
- CUDA kernel implementation for matrix operations
- Optimized inference on Jetson TX2
- Performance benchmarking vs PyTorch

### Architecture Generalization
- ✓ Dynamic multilayer fully connected networks
- Convolutional and recurrent layer support
- Generalized weight loading from arbitrary PyTorch models
- Automated model architecture export

### Advanced Features
- ✓ Support for MNIST and CIFAR-10 datasets
- Visualization tools for test data and predictions
- PyTorch-to-inference validation pipeline
- Quantization (FP16, INT8)
- Dynamic batch size support
