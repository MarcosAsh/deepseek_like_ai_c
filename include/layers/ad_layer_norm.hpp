#pragma once
#include "autodiff.hpp"

// AD-aware Layer Normalization
class ADLayerNorm {
public:
    ADLayerNorm(int dim, float eps = 1e-5f);
    // x: [dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    int dim;
    float eps;
    std::shared_ptr<ADTensor> gamma;
    std::shared_ptr<ADTensor> beta;
};