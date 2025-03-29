#pragma once
#include "tensor.hpp"

class Linear {
public:
    Tensor weights;
    Tensor bias;

    Linear(int input_size, int output_size);
    Tensor forward(const Tensor& input);
};
