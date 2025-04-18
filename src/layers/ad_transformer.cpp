#include "layers/ad_transformer.hpp"

// AD Transformer Block
ADTransformerBlock::ADTransformerBlock(int embed_dim, int hidden_dim, int n_heads)
    : ln1(embed_dim),
      mha(embed_dim, n_heads),
      ln2(embed_dim),
      ff(embed_dim, hidden_dim) {}

std::shared_ptr<ADTensor> ADTransformerBlock::forward(
    const std::shared_ptr<ADTensor>& x) {
    // Pre-norm & Attention
    auto x1 = ln1.forward(x);
    // DEBUG: pre-attn shapes
    std::cout << "DEBUG Block pre-attn x shape: [" << x->val.rows
              << "x" << x->val.cols << "]\n";
    std::cout << "DEBUG Block pre-attn x1 shape: [" << x1->val.rows
              << "x" << x1->val.cols << "]\n";
    auto a = mha.forward(x1);
    // DEBUG: attention output shape: [rows x cols]
    std::cout << "DEBUG Block attn a shape: [" << a->val.rows
              << "x" << a->val.cols << "]\n";
    std::cout << "DEBUG Block residual x shape: [" << x->val.rows
              << "x" << x->val.cols << "]\n";
    auto x2 = add(a, x);
    // Pre-norm & FeedForward
    auto x3 = ln2.forward(x2);
    auto f = ff.forward(x3);
    auto x4 = add(f, x2);
    return x4;
}

// AD Transformer
ADTransformer::ADTransformer(int num_layers, int embed_dim, int hidden_dim, int n_heads) {
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(embed_dim, hidden_dim, n_heads);
    }
}

std::shared_ptr<ADTensor> ADTransformer::forward(const std::shared_ptr<ADTensor>& x) {
    auto out = x;
    for (auto& block : blocks) {
        out = block.forward(out);
    }
    return out;
}