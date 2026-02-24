#pragma once
#include "autodiff.hpp"

// Root Mean Square Layer Normalization as used in LLaMA / DeepSeek
// Faster than LayerNorm: skips mean centering, only does RMS scaling
// RMSNorm(x) = x / RMS(x) * gamma, where RMS(x) = sqrt(mean(x^2) + eps)
class ADRMSNorm {
public:
    ADRMSNorm(int dim, float eps = 1e-6f);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    int dim_;
    float eps_;
    std::shared_ptr<ADTensor> gamma;

    // Cached tensors
    mutable Tensor cached_ones_row{1, 1};
    mutable Tensor cached_ones_col{1, 1};
    mutable Tensor cached_ones_cols{1, 1};
    mutable int cached_cols = -1;
};
