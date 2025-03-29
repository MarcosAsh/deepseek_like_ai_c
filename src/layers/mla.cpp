#include "layers/mla.hpp"
#include <iostream>

Tensor MLA::forward(const Tensor& input) {
    std::cout << "MLA layer forward" << std::endl;
    return input;
}
