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
    // Cached ones tensors
    mutable Tensor cached_ones_row{1, 1};   // [1 x dim]
    mutable Tensor cached_ones_col{1, 1};   // [dim x 1]
    mutable Tensor cached_ones_cols{1, 1};  // [1 x cols]
    mutable int cached_cols = -1;
};