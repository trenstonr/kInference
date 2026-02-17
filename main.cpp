#include "include/tensor.hpp"
#include "include/engine.hpp"
#include "include/inferencer.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Example: 3-layer CIFAR-10 network (3072 -> 256 -> 128 -> 10)
    // Activations: relu, relu, softmax

    std::vector<std::string> weights = {
        "exported_data_cifar10/weights_fc1_weight.bin",
        "exported_data_cifar10/weights_fc2_weight.bin",
        "exported_data_cifar10/weights_fc3_weight.bin"
    };
    std::vector<std::string> biases = {
        "exported_data_cifar10/weights_fc1_bias.bin",
        "exported_data_cifar10/weights_fc2_bias.bin",
        "exported_data_cifar10/weights_fc3_bias.bin"
    };
    
    Inferencer inferencer(weights, biases);
    
    std::cout << "CIFAR-10 Network has " << inferencer.getNumLayers() << " layers\n\n";
    
    // Load and run inference on test image 0
    Tensor image("exported_data_cifar10/test_image_0.bin");
    Tensor expected("exported_data_cifar10/expected_output_0.bin");
    
    // Run inference
    Tensor prediction = inferencer.infer(image);
    
    // Load results
    std::cout << "Test Image 0:\n";
    std::cout << "  Prediction: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(4) << prediction.data()[i] << " ";
    }
    std::cout << "\n";
    
    // Compare to PyTorch results
    std::cout << "  Expected:   ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(4) << expected.data()[i] << " ";
    }
    std::cout << "\n";
    
    return 0;
}

