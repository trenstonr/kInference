#include "include/tensor.hpp"
#include "include/engine.hpp"

int main() {
    Tensor a("exported_data/weights_fc1_weight.bin");
    a.print();
    return 0;
}

