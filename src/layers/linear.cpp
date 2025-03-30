#include "layers/linear.hpp"
#include <iostream>

Linear::Linear(int input_size, int output_size)
    : weights(output_size), bias(output_size) {
    weights.fill(0.1f); // Dummy init
    bias.fill(0.2f);    // Dummy bias
    }


Tensor Linear::forward(const Tensor& input) {
    assert(input.rows == weights.cols);
    Tensor output = weights.matmul(input);

    // Add bias
    for (int i = 0 ; i < output.rows; ++i) {
        output.data[i] += bias.data[i];
    }

    output.print("Linear output");
    return output;
}