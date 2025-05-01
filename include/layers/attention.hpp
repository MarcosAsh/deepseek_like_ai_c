#pragma once
#include "tensor.hpp"
// Multi-head self-attention layer
class MultiHeadAttention {
public:
    // embed_dim: model dimension
    // num_heads: number of attention heads
    // embed_dim: model dimension, num_heads: number of attention heads
    // causal: if true, apply autoregressive (causal) masking
    MultiHeadAttention(int embed_dim, int num_heads, bool causal = false);
    // input: [embed_dim x seq_len], returns [embed_dim x seq_len]
    Tensor forward(const Tensor& input) const;

private:
    int embed_dim;
    int num_heads;
    int head_dim;
    // whether to apply causal (autoregressive) masking
    bool causal;
    Tensor W_q;
    Tensor W_k;
    Tensor W_v;
    Tensor W_o;
};