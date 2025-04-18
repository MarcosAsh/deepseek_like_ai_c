#pragma once
#include "tensor.hpp"
#include "layers/attention.hpp"
#include "layers/layer_norm.hpp"
#include "layers/feed_forward.hpp"
#include "layers/dropout.hpp"
#include <vector>

class TransformerBlock {
    LayerNorm ln1, ln2;
    MultiHeadAttention mha;
    FeedForward ff;
    Dropout dropout1, dropout2;
    
public:
    // input_dim: model dimension, hidden_dim: feed-forward dimension,
    // n_heads: number of attention heads
    // input_dim: model dimension, hidden_dim: feed-forward inner dimension,
    // n_heads: number of attention heads
    TransformerBlock(int input_dim, int hidden_dim, int n_heads);
    // training: apply dropout when true (default false)
    Tensor forward(const Tensor& input, bool training = false);
};

class Transformer {
public:
    Transformer(int num_layers, int input_dim, int hidden_dim,
               int n_heads);
    // training: propagate dropout flag (default false)
    Tensor forward(const Tensor& input, bool training = false);

private:
    std::vector<TransformerBlock> blocks;
};