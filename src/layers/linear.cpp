#include "layers/linear.hpp"
#include <random>

Linear::Linear(int input_size, int output_size)
    : weights(output_size, input_size), bias(output_size, 1) {
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float range = std::sqrt(6.0f / (input_size + output_size));
    std::uniform_real_distribution<float> dist(-range, range);
    
    for (float& w : weights.data) w = dist(gen);
    bias.fill(0.1f);  // Small constant bias
}

Tensor Linear::forward(const Tensor& input) const {
    assert(input.cols == 1 && "Linear layer expects column vector");
    assert(input.rows == weights.cols && "Input dimension mismatch");
    
    Tensor output = weights.matmul(input);
    
    // Add bias
    for (int i = 0; i < output.rows; ++i) {
        output.data[i] += bias.data[i];
    }
    
    return output;
}