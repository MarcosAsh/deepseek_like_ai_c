#pragma once
#include "autodiff.hpp"
#include <limits>

class ADMultiHeadAttention {
public:
    ADMultiHeadAttention(int embed_dim, int num_heads, bool causal = true);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    int embed_dim;
    int num_heads;
    int head_dim;
    bool causal;
    // ALiBi slopes for each head
    std::vector<float> alibi_slopes;
    std::shared_ptr<ADTensor> W_q, W_k, W_v, W_o;
};