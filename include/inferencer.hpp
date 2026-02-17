#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include "tensor.hpp"
#include <vector>
#include <string>

class Inferencer {
public:
    // Inferencer();
    
    Inferencer(const std::vector<std::string>& weights_paths, 
               const std::vector<std::string>& bias_paths);

    Tensor infer(const Tensor& image);
    
    int getNumLayers() const;

private:
    struct Layer {
        Tensor weight;
        Tensor bias;
    };
    
    std::vector<Layer> layers;
};

#endif
