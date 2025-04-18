#pragma once
#include "autodiff.hpp"

// AD-aware Multi-Head Self-Attention
class ADMultiHeadAttention {
public:
    ADMultiHeadAttention(int embed_dim, int num_heads);
    // input: [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    int embed_dim;
    int num_heads;
    int head_dim;
    std::shared_ptr<ADTensor> W_q, W_k, W_v, W_o;
};