#pragma once
#include "tensor.hpp"
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int embed_dim);
    Tensor forward(const std::vector<int>& tokens) const;

    Tensor weights; // [embed_dim x vocab_size]
};