# C++ Inference Engine

A lightweight neural network inference engine written in C++ for deployment on embedded systems, particularly the NVIDIA Jetson TX2.

## Overview

Custom inference engine for deploying PyTorch models in C++. Currently implements a 2-layer fully connected network (784→128→10) trained on MNIST, with plans for CUDA acceleration and generalized architecture support.

## Architecture

### Tensor Class (`include/tensor.hpp`, `src/tensor.cpp`)
- N-dimensional array container
- Binary file loading for PyTorch-exported weights
- Memory management and reshape operations

### Engine Class (`include/engine.hpp`, `src/engine.cpp`)
- Matrix operations: matmul, element-wise add/multiply
- Activation functions: ReLU, Sigmoid, Softmax
- Core computational primitives

### Inferencer Class (`include/inferencer.hpp`, `src/inferencer.cpp`)
- Implements 2-layer FC network: FC→ReLU→FC→Softmax
- Loads MNIST model weights from binary files
- Executes forward pass

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
│   └── train_mnist.py      # PyTorch script to export MNIST weights/data
└── main.cpp                # Example MNIST inference
```

## Usage

Train the MNIST model and export weights:

```bash
cd python
python train_mnist.py
cd ..
```

This creates `exported_data/` with weight matrices, bias vectors, test images, and expected outputs.

Compile and run the C++ inference:

```bash
g++ -std=c++17 main.cpp src/* -I./include -o inference_engine
./inference_engine
```

Example output:

```
Test Image 0:
  Prediction: 0.0001 0.0000 0.0012 0.9985 0.0000 0.0001 0.0000 0.0000 0.0001 0.0000
  Expected:   0.0001 0.0000 0.0012 0.9985 0.0000 0.0001 0.0000 0.0000 0.0001 0.0000
```

## Roadmap

### CUDA Acceleration
- CUDA kernel implementation for matrix operations
- Optimized inference on Jetson TX2
- Performance benchmarking vs PyTorch

### Architecture Generalization
- Dynamic layer configuration
- Convolutional and recurrent layer support
- Generalized weight loading from arbitrary PyTorch models
- Automated model architecture export

### Advanced Features
- PyTorch-to-inference validation pipeline
- Testing on complex datasets beyond MNIST
- Quantization (FP16, INT8)
- Dynamic batch size support
