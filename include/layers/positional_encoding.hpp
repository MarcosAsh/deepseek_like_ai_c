#pragma once
#include "tensor.hpp"

class PositionalEncoding {
public:
    PositionalEncoding(int embed_dim, int max_len = 512);
    Tensor forward(int seq_len) const;

private:
    int embed_dim;
    int max_len;
    Tensor pe; // [embed_dim x max_len]
};