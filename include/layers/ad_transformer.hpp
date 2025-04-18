#pragma once
#include <vector>
#include "autodiff.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_multi_head_attention.hpp"
#include "layers/ad_feed_forward.hpp"

// AD-aware Transformer block: pre-norm, attention, residual, pre-norm, FF, residual
class ADTransformerBlock {
public:
    // embed_dim: model dimension, hidden_dim: FF inner dimension, n_heads: number of heads
    ADTransformerBlock(int embed_dim, int hidden_dim, int n_heads);
    // x: [embed_dim x seq_len]
    // returns: [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    ADLayerNorm ln1;
    ADMultiHeadAttention mha;
    ADLayerNorm ln2;
    ADFeedForward ff;
};

// AD-aware Transformer: sequence of ADTransformerBlock
class ADTransformer {
public:
    // num_layers: number of blocks
    ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads);
    // x: [embed_dim x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    std::vector<ADTransformerBlock> blocks;
};