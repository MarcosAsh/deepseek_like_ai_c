#include "layers/ad_transformer.hpp"

// AD Transformer Block
ADTransformerBlock::ADTransformerBlock(int embed_dim, int hidden_dim, int n_heads,
                                       bool use_moe_, int num_experts, int moe_top_k)
    : ln1(embed_dim),
      mha(embed_dim, n_heads),
      ln2(embed_dim),
      ff(embed_dim, hidden_dim),
      use_moe(use_moe_) {
    if (use_moe) {
        moe = std::make_unique<ADMoE>(embed_dim, hidden_dim, num_experts, moe_top_k);
    }
}

std::shared_ptr<ADTensor> ADTransformerBlock::forward(
    const std::shared_ptr<ADTensor>& x,
    std::shared_ptr<ADTensor>* aux_loss) {
    // Pre-norm & Attention
    auto x1 = ln1.forward(x);
    auto a = mha.forward(x1);
    auto x2 = add(a, x);
    // Pre-norm & FeedForward (or MoE)
    auto x3 = ln2.forward(x2);
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
    } else {
        f = ff.forward(x3);
    }
    auto x4 = add(f, x2);
    return x4;
}

// AD Transformer
ADTransformer::ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads,
                             bool use_moe, int num_experts, int moe_top_k) {
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(embed_dim, hidden_dim, n_heads, use_moe, num_experts, moe_top_k);
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
