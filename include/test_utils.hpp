#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <vector>

void assert_close(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float eps = 1e-5f
);

#endif

