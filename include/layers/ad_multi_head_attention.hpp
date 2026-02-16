#pragma once
#include "autodiff.hpp"
#include <limits>

// AD-aware Multi-Head Self-Attention
class ADMultiHeadAttention {
public:
    // causal: if true, apply autoregressive masking (future positions get -inf)
    ADMultiHeadAttention(int embed_dim, int num_heads, bool causal = true);
    // input: [embed_dim x seq_len]
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