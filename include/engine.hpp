#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "tensor.hpp"

class Engine {
public:
    // element-wise
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor multiply(const Tensor& a, const Tensor& b);
    static Tensor relu(const Tensor& x);
    static Tensor sigmoid(const Tensor& x);

    // matrix operations
    static Tensor matmul(const Tensor& a, const Tensor& b);

    // other
    static Tensor softmax(const Tensor& x);

private:
    static void check_same_shape(const Tensor& a, const Tensor& b);
};

#endif

