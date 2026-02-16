#pragma once
#include <vector>
#include <memory>
#include "autodiff.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_multi_head_attention.hpp"
#include "layers/ad_feed_forward.hpp"
#include "layers/ad_moe.hpp"

// AD-aware Transformer block: pre-norm, attention, residual, pre-norm, FF/MoE, residual
class ADTransformerBlock {
public:
    // embed_dim, hidden_dim, n_heads: standard params
    // use_moe: if true, replace FF with MoE layer
    // num_experts, moe_top_k: MoE configuration
    ADTransformerBlock(int embed_dim, int hidden_dim, int n_heads,
                       bool use_moe = false, int num_experts = 4, int moe_top_k = 2);
    // x: [embed_dim x seq_len]
    // returns: [embed_dim x seq_len]
    // aux_loss: accumulated MoE auxiliary loss (0 if no MoE)
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x,
                                       std::shared_ptr<ADTensor>* aux_loss = nullptr);

private:
    ADLayerNorm ln1;
    ADMultiHeadAttention mha;
    ADLayerNorm ln2;
    ADFeedForward ff;
    bool use_moe;
    std::unique_ptr<ADMoE> moe;
};

// AD-aware Transformer: sequence of ADTransformerBlock
class ADTransformer {
public:
    // use_moe, num_experts, moe_top_k: optional MoE config
    ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads,
                  bool use_moe = false, int num_experts = 4, int moe_top_k = 2);
    // Returns transformer output and accumulated MoE aux loss (nullptr if no MoE)
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x,
                                       std::shared_ptr<ADTensor>* aux_loss = nullptr);

private:
    std::vector<ADTransformerBlock> blocks;
};