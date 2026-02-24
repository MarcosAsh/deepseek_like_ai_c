#pragma once
#include <vector>
#include <memory>
#include "autodiff.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_rmsnorm.hpp"
#include "layers/ad_multi_head_attention.hpp"
#include "layers/ad_feed_forward.hpp"
#include "layers/ad_swiglu.hpp"
#include "layers/ad_moe.hpp"

struct TransformerConfig {
    int embed_dim = 64;
    int hidden_dim = 64;
    int n_heads = 4;
    int num_layers = 3;
    bool use_moe = false;
    int num_experts = 4;
    int moe_top_k = 2;
    bool use_rmsnorm = false;
    bool use_swiglu = false;
    bool use_rope = false;
};

class ADTransformerBlock {
public:
    ADTransformerBlock(const TransformerConfig& cfg);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x,
                                       std::shared_ptr<ADTensor>* aux_loss = nullptr);

private:
    // Normalization (either LayerNorm or RMSNorm)
    std::unique_ptr<ADLayerNorm> ln1;
    std::unique_ptr<ADLayerNorm> ln2;
    std::unique_ptr<ADRMSNorm> rn1;
    std::unique_ptr<ADRMSNorm> rn2;
    bool use_rmsnorm;

    ADMultiHeadAttention mha;

    // FFN (either GELU FFN, SwiGLU, or MoE)
    std::unique_ptr<ADFeedForward> ff;
    std::unique_ptr<ADSwiGLU> swiglu;
    bool use_moe;
    std::unique_ptr<ADMoE> moe;

    std::shared_ptr<ADTensor> norm1(const std::shared_ptr<ADTensor>& x);
    std::shared_ptr<ADTensor> norm2(const std::shared_ptr<ADTensor>& x);
};

class ADTransformer {
public:
    // Legacy constructor for backward compatibility
    ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads,
                  bool use_moe = false, int num_experts = 4, int moe_top_k = 2);
    // New config-based constructor
    explicit ADTransformer(const TransformerConfig& cfg);

    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x,
                                       std::shared_ptr<ADTensor>* aux_loss = nullptr);

private:
    std::vector<ADTransformerBlock> blocks;
};
