#pragma once
#include "tensor.hpp"
// Multi-head self-attention layer
class MultiHeadAttention {
public:
    // embed_dim: model dimension
    // num_heads: number of attention heads
    // embed_dim: model dimension, num_heads: number of attention heads
    // causal: if true, apply autoregressive (causal) masking
    // dropout_prob: probability of dropping attention weights after softmax
    MultiHeadAttention(int embed_dim, int num_heads, bool causal = false,
                       float dropout_prob = 0.0f);
    // input: [embed_dim x seq_len], returns [embed_dim x seq_len]
    Tensor forward(const Tensor& input) const;

private:
    int embed_dim;
    int num_heads;
    int head_dim;
    // whether to apply causal (autoregressive) masking
    bool causal;
    // dropout probability applied to attention weights
    float dropout_prob;
    Tensor W_q;
    Tensor W_k;
    Tensor W_v;
    Tensor W_o;
};