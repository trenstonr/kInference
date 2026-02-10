#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include "tensor.hpp"

class Inferencer {
public:
    Inferencer();

    Tensor infer(const Tensor& image);

private:
    Tensor fc1_weight;
    Tensor fc1_bias;
    Tensor fc2_weight;
    Tensor fc2_bias;
};

#endif
