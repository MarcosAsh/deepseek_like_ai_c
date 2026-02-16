#pragma once
#include "tensor.hpp"
// Multi-head self-attention layer with optional KV cache for autoregressive generation
class MultiHeadAttention {
public:
    // embed_dim: model dimension, num_heads: number of attention heads
    // causal: if true, apply autoregressive (causal) masking
    // dropout_prob: probability of dropping attention weights after softmax
    MultiHeadAttention(int embed_dim, int num_heads, bool causal = false,
                       float dropout_prob = 0.0f);
    // input: [embed_dim x seq_len], returns [embed_dim x seq_len]
    // training: apply dropout when true (default false for inference)
    // use_cache: when true, append K/V to internal cache and attend over full cache
    Tensor forward(const Tensor& input, bool training = false, bool use_cache = false);
    // Clear the KV cache (call before starting a new sequence)
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