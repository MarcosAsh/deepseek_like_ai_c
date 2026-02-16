#pragma once
#include "tensor.hpp"
#include "layers/attention.hpp"
#include "layers/layer_norm.hpp"
#include "layers/feed_forward.hpp"
#include "layers/dropout.hpp"
#include <vector>

class TransformerBlock {
public:
    LayerNorm ln1;
    MultiHeadAttention mha;
    LayerNorm ln2;
    FeedForward ff;
    Dropout dropout1, dropout2;

    TransformerBlock(int input_dim, int hidden_dim, int n_heads);
    Tensor forward(const Tensor& input, bool training = false, bool use_cache = false);
    void clear_cache();
};

class Transformer {
public:
    Transformer(int num_layers, int input_dim, int hidden_dim,
               int n_heads);
    Tensor forward(const Tensor& input, bool training = false, bool use_cache = false);
    void clear_cache();

    std::vector<TransformerBlock> blocks;
};