#include "../include/inferencer.hpp"
#include "../include/engine.hpp"
#include <stdexcept>

// Inferencer::Inferencer() {
//     // Default constructor: load the 2-layer network from exported_data
//     std::vector<std::string> weights = {
//         "exported_data/weights_fc1_weight.bin",
//         "exported_data/weights_fc2_weight.bin"
//     };
//     std::vector<std::string> biases = {
//         "exported_data/weights_fc1_bias.bin",
//         "exported_data/weights_fc2_bias.bin"
//     };
    
//     // Initialize with default paths
//     for (size_t i = 0; i < weights.size(); ++i) {
//         Layer layer;
//         layer.weight = Tensor(weights[i]);
//         layer.bias = Tensor(biases[i]);
//         layers.push_back(layer);
//     }
// }

Inferencer::Inferencer(const std::vector<std::string>& weights_paths, const std::vector<std::string>& bias_paths) {
    if (weights_paths.size() != bias_paths.size()) {
        throw std::runtime_error("Number of weight and bias paths must match");
    }
    
    // Load all layers
    for (size_t i = 0; i < weights_paths.size(); ++i) {
        Layer layer;
        layer.weight = Tensor(weights_paths[i]);
        layer.bias = Tensor(bias_paths[i]);
        layers.push_back(layer);
    }
}

Tensor Inferencer::infer(const Tensor& image) {
    Tensor current = image;
    
    // Process through each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        bool is_last_layer = (i == layers.size() - 1);
        
        // Matrix multiplication with weight
        current = Engine::matmul(current, layer.weight);
        
        // Reshape bias if needed and add it
        Tensor bias = Tensor(layer.bias.data(), {1, layer.bias.shape()[0]});
        current = Engine::add(current, bias);
        
        // Apply activation: relu for hidden layers, softmax for output layer
        if (is_last_layer) {
            current = Engine::softmax(current);
        } else {
            current = Engine::relu(current);
        }
    }
    
    return current;
}

int Inferencer::getNumLayers() const {
    return layers.size();
}
