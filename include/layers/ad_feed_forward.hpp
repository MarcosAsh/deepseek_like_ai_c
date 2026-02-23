#pragma once
#include "autodiff.hpp"

class ADFeedForward {
public:
    ADFeedForward(int embed_dim, int hidden_dim);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    std::shared_ptr<ADTensor> W1, b1;
    std::shared_ptr<ADTensor> W2, b2;
    // Cached ones tensors for bias broadcast
    mutable Tensor cached_ones1{1, 1};
    mutable Tensor cached_ones2{1, 1};
    mutable int cached_seq_len = -1;
};