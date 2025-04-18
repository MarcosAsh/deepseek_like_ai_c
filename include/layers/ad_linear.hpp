#pragma once
#include "autodiff.hpp"

// AD-aware Linear layer: y = W * x + b
class ADLinear {
public:
    // input_dim: number of input features
    // output_dim: number of output features
    ADLinear(int input_dim, int output_dim);
    // Forward pass: x [input_dim x seq_len] -> [output_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x) const;

private:
    std::shared_ptr<ADTensor> W;  // [output_dim x input_dim]
    std::shared_ptr<ADTensor> b;  // [output_dim x 1]
};