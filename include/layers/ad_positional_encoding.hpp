#pragma once
#include "autodiff.hpp"

// AD-aware sinusoidal positional encoding
class ADPositionalEncoding {
public:
    ADPositionalEncoding(int embed_dim, int max_len = 512);
    // returns [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(int seq_len) const;

private:
    int embed_dim;
    int max_len;
    Tensor pe; // precomputed [embed_dim x max_len]
};