#pragma once
#include "autodiff.hpp"

// AD-aware learned positional embedding
class ADPositionalEncoding {
public:
    // embed_dim: embedding dimension, max_len: max sequence length
    ADPositionalEncoding(int embed_dim, int max_len = 512);
    // returns positional embeddings [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(int seq_len) const;

private:
    int embed_dim;
    int max_len;
    std::shared_ptr<ADTensor> pweights; // learnable [embed_dim x max_len]
};