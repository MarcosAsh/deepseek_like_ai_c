#pragma once
#include "autodiff.hpp"

// AD-aware position-wise Feed-Forward network with GELU activation
class ADFeedForward {
public:
    // embed_dim: model dimension; hidden_dim: inner FF dimension
    ADFeedForward(int embed_dim, int hidden_dim);
    // Forward: x [embed_dim x seq_len] -> y [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    std::shared_ptr<ADTensor> W1, b1;
    std::shared_ptr<ADTensor> W2, b2;
};