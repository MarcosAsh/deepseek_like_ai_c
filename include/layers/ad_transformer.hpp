#pragma once
#include <vector>
#include <memory>
#include "autodiff.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_multi_head_attention.hpp"
#include "layers/ad_feed_forward.hpp"
#include "layers/ad_moe.hpp"

class ADTransformerBlock {
public:
    ADTransformerBlock(int embed_dim, int hidden_dim, int n_heads,
                       bool use_moe = false, int num_experts = 4, int moe_top_k = 2);
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

class ADTransformer {
public:
    ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads,
                  bool use_moe = false, int num_experts = 4, int moe_top_k = 2);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x,
                                       std::shared_ptr<ADTensor>* aux_loss = nullptr);

private:
    std::vector<ADTransformerBlock> blocks;
};