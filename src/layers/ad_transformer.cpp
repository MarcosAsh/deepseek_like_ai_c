#include "layers/ad_transformer.hpp"

// AD Transformer Block
ADTransformerBlock::ADTransformerBlock(const TransformerConfig& cfg)
    : use_rmsnorm(cfg.use_rmsnorm),
      mha(cfg.embed_dim, cfg.n_heads),
      use_moe(cfg.use_moe) {
    // Normalization layers
    if (cfg.use_rmsnorm) {
        rn1 = std::make_unique<ADRMSNorm>(cfg.embed_dim);
        rn2 = std::make_unique<ADRMSNorm>(cfg.embed_dim);
    } else {
        ln1 = std::make_unique<ADLayerNorm>(cfg.embed_dim);
        ln2 = std::make_unique<ADLayerNorm>(cfg.embed_dim);
    }

    // Feed-forward variant
    if (cfg.use_moe) {
        moe = std::make_unique<ADMoE>(cfg.embed_dim, cfg.hidden_dim,
                                       cfg.num_experts, cfg.moe_top_k);
    } else if (cfg.use_swiglu) {
        swiglu = std::make_unique<ADSwiGLU>(cfg.embed_dim, cfg.hidden_dim);
    } else {
        ff = std::make_unique<ADFeedForward>(cfg.embed_dim, cfg.hidden_dim);
    }
}

std::shared_ptr<ADTensor> ADTransformerBlock::norm1(const std::shared_ptr<ADTensor>& x) {
    if (use_rmsnorm) return rn1->forward(x);
    return ln1->forward(x);
}

std::shared_ptr<ADTensor> ADTransformerBlock::norm2(const std::shared_ptr<ADTensor>& x) {
    if (use_rmsnorm) return rn2->forward(x);
    return ln2->forward(x);
}

std::shared_ptr<ADTensor> ADTransformerBlock::forward(
    const std::shared_ptr<ADTensor>& x,
    std::shared_ptr<ADTensor>* aux_loss) {
    // Pre-norm & Attention
    auto x1 = norm1(x);
    auto a = mha.forward(x1);
    auto x2 = add(a, x);
    // Pre-norm & FeedForward (or MoE or SwiGLU)
    auto x3 = norm2(x2);
    std::shared_ptr<ADTensor> f;
    if (use_moe && moe) {
        auto moe_out = moe->forward(x3);
        f = moe_out.output;
        if (aux_loss) {
            if (*aux_loss) {
                *aux_loss = add(*aux_loss, moe_out.aux_loss);
            } else {
                *aux_loss = moe_out.aux_loss;
            }
        }
    } else if (swiglu) {
        f = swiglu->forward(x3);
    } else {
        f = ff->forward(x3);
    }
    auto x4 = add(f, x2);
    return x4;
}

// Legacy constructor
ADTransformer::ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads,
                             bool use_moe, int num_experts, int moe_top_k) {
    TransformerConfig cfg;
    cfg.embed_dim = embed_dim;
    cfg.hidden_dim = hidden_dim;
    cfg.n_heads = n_heads;
    cfg.num_layers = num_layers;
    cfg.use_moe = use_moe;
    cfg.num_experts = num_experts;
    cfg.moe_top_k = moe_top_k;
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(cfg);
    }
}

// Config-based constructor
ADTransformer::ADTransformer(const TransformerConfig& cfg) {
    for (int i = 0; i < cfg.num_layers; ++i) {
        blocks.emplace_back(cfg);
    }
}

std::shared_ptr<ADTensor> ADTransformer::forward(
    const std::shared_ptr<ADTensor>& x,
    std::shared_ptr<ADTensor>* aux_loss) {
    auto out = x;
    for (auto& block : blocks) {
        out = block.forward(out, aux_loss);
    }
    return out;
}
