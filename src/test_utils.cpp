#include "../include/test_utils.hpp"
#include <cmath>
#include <stdexcept>

void assert_close(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float eps
) {
    if (a.size() != b.size())
        throw std::runtime_error("size mismatch");

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps)
            throw std::runtime_error("value mismatch at index " + std::to_string(i));
    }
}

