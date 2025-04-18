#pragma once
#include "autodiff.hpp"
#include <vector>

// AD-aware token embedding: lookup via one-hot matmul
class ADEmbedding {
public:
    ADEmbedding(int vocab_size, int embed_dim);
    // tokens: sequence of token IDs (length seq_len)
    // returns: [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::vector<int>& tokens) const;

private:
    int vocab_size;
    int embed_dim;
    std::shared_ptr<ADTensor> weights; // [embed_dim x vocab_size]
};