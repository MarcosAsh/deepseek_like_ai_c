#pragma once
#include "tensor.hpp"
class MultiHeadAttention {
public:
    MultiHeadAttention(int embed_dim, int num_heads, bool causal = false,
                       float dropout_prob = 0.0f);
    Tensor forward(const Tensor& input, bool training = false, bool use_cache = false);
    void clear_cache();

    int embed_dim;
    int num_heads;
    int head_dim;
    bool causal;
    float dropout_prob;
    Tensor W_q;
    Tensor W_k;
    Tensor W_v;
    Tensor W_o;
    // KV cache: [embed_dim x cached_len]
    Tensor k_cache;
    Tensor v_cache;
};