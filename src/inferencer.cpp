#include "../include/inferencer.hpp"
#include "../include/engine.hpp"

Inferencer::Inferencer() {
    fc1_weight = Tensor("exported_data/weights_fc1_weight.bin");
    fc1_bias = Tensor("exported_data/weights_fc1_bias.bin");
    fc2_weight = Tensor("exported_data/weights_fc2_weight.bin");
    fc2_bias = Tensor("exported_data/weights_fc2_bias.bin");
}

Tensor Inferencer::infer(const Tensor& image) {
    // First layer: matmul + bias + relu
    Tensor hidden = Engine::matmul(image, fc1_weight);
    Tensor bias1 = Tensor(fc1_bias.data(), {1, fc1_bias.shape()[0]}); // reshape bias from [x] to [1, x]
    hidden = Engine::add(hidden, bias1);
    hidden = Engine::relu(hidden);
    
    // Second layer: matmul + bias + softmax
    Tensor output = Engine::matmul(hidden, fc2_weight);
    Tensor bias2 = Tensor(fc2_bias.data(), {1, fc2_bias.shape()[0]});
    output = Engine::add(output, bias2);
    output = Engine::softmax(output);
    
    return output;
}
