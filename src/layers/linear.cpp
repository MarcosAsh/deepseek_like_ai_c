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
    assert(input.rows == weights.cols && "Input dimension mismatch");

    Tensor output = weights.matmul(input);

    // Add bias (broadcast across columns)
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.data[i * output.cols + j] += bias.data[i];
        }
    }

    return output;
}