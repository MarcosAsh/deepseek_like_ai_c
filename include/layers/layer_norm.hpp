#pragma once
#include "tensor.hpp"

class LayerNorm {
public:
    LayerNorm(int dim, float eps = 1e-5f);
    Tensor forward(const Tensor& input) const;

    int dim;
    float eps;
    Tensor gamma; // [dim x 1]
    Tensor beta;  // [dim x 1]
};