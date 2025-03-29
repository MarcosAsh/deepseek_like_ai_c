#include "layers/linear.hpp"
#include <iostream>

Linear::Linear(int input_size, int output_size)
    : weights(output_size), bias(output_size) {}

Tensor Linear::forward(const Tensor& input) {
    std::cout << "Linear layer forward" << std::endl;
    return Tensor(weights.size); // Dummy output
}
