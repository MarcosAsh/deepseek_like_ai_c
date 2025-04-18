#pragma once
#include "tensor.hpp"
#include <vector>

// Token embedding layer: maps token IDs to dense vectors
class Embedding {
public:
    // vocab_size: number of tokens in vocabulary
    // embed_dim: dimensionality of embedding vectors
    Embedding(int vocab_size, int embed_dim);
    // Forward pass: tokens -> [embed_dim x seq_len] tensor
    Tensor forward(const std::vector<int>& tokens) const;

private:
    Tensor weights; // [embed_dim x vocab_size]
};