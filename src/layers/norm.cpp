#include "layers/norm.hpp"
#include <iostream>

Tensor Norm::forward(const Tensor& input) {
    std::cout << "Norm layer forward" << std::endl;
    return input;
}
