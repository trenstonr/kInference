#include "../include/engine.hpp"
#include <cmath>
#include <stdexcept>

void Engine::check_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("shape mismatch");
}

Tensor Engine::add(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);

    std::vector<float> out(a.size());
    for (int i = 0; i < a.size(); ++i)
        out[i] = a.data()[i] + b.data()[i];

    return Tensor(out, a.shape());
}

Tensor Engine::multiply(const Tensor& a, const Tensor& b) {
    check_same_shape(a, b);

    std::vector<float> out(a.size());
    for (int i = 0; i < a.size(); ++i)
        out[i] = a.data()[i] * b.data()[i];

    return Tensor(out, a.shape());
}

Tensor Engine::relu(const Tensor& x) {
    std::vector<float> out(x.size());
    for (int i = 0; i < x.size(); ++i)
        out[i] = std::max(0.0f, x.data()[i]);

    return Tensor(out, x.shape());
}

Tensor Engine::sigmoid(const Tensor& x) {
    std::vector<float> out(x.size());
    for (int i = 0; i < x.size(); ++i)
        out[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));

    return Tensor(out, x.shape());
}

Tensor Engine::matmul(const Tensor& a, const Tensor& b) {
    auto as = a.shape();
    auto bs = b.shape();

    if (as.size() != 2 || bs.size() != 2 || as[1] != bs[0])
        throw std::runtime_error("invalid matmul shapes");

    int m = as[0], k = as[1], n = bs[1];
    std::vector<float> out(m * n, 0.0f);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int p = 0; p < k; ++p)
                out[i * n + j] += a.data()[i * k + p] * b.data()[p * n + j];

    return Tensor(out, {m, n});
}

Tensor Engine::softmax(const Tensor& x) {
    if (x.shape().size() != 2)
        throw std::runtime_error("softmax expects 2D");

    int rows = x.shape()[0];
    int cols = x.shape()[1];

    std::vector<float> out(x.size());

    for (int r = 0; r < rows; ++r) {
        float maxv = -INFINITY;
        for (int c = 0; c < cols; ++c)
            maxv = std::max(maxv, x.data()[r * cols + c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            out[r * cols + c] = std::exp(x.data()[r * cols + c] - maxv);
            sum += out[r * cols + c];
        }

        for (int c = 0; c < cols; ++c)
            out[r * cols + c] /= sum;
    }

    return Tensor(out, x.shape());
}

