#pragma once
#include "tensor.hpp"

// Layer normalization with learnable gain and bias
class LayerNorm {
public:
    // dim: number of features (embed dimension), eps: small constant for stability
    LayerNorm(int dim, float eps = 1e-5f);
    // input: [dim x seq_len], returns normalized tensor of same shape
    Tensor forward(const Tensor& input) const;

    int dim;
    float eps;
    Tensor gamma; // [dim x 1]
    Tensor beta;  // [dim x 1]
};