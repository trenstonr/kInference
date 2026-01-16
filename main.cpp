#include "include/tensor.hpp"
#include "include/engine.hpp"

int main() {
    Tensor a({1,2,3,4}, {2,2});
    Tensor b({5,6,7,8}, {2,2});

    Tensor c = Engine::matmul(a, b);
    c.print();

    Tensor d = Engine::relu(c);
    d.print();

    return 0;
}

