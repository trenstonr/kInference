#include "include/tensor.hpp"
#include "include/engine.hpp"
#include "include/inferencer.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Create inferencer (loads weights and biases)
    Inferencer inferencer;
    
    // Load and run inference on test image 0
    Tensor image("exported_data/test_image_0.bin");
    Tensor expected("exported_data/expected_output_0.bin");
    
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

