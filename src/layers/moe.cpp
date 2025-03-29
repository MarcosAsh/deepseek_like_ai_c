#include "layers/moe.hpp"
#include <iostream>

Tensor MoE::forward(const Tensor& input) {
    std::cout << "MoE layer forward" << std::endl;
    return input;
}
