#pragma once
#include "tensor.hpp"

// Sinusoidal positional encoding
class PositionalEncoding {
public:
    // embed_dim: dimension of embeddings
    // max_len: maximum sequence length
    PositionalEncoding(int embed_dim, int max_len = 512);
    // Generate positional encodings for sequence length seq_len
    Tensor forward(int seq_len) const;

private:
    int embed_dim;
    int max_len;
    Tensor pe; // [embed_dim x max_len]
};