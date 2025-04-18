#include "../include/transformer.hpp"

TransformerBlock::TransformerBlock(int input_dim, int hidden_dim,
                                   int n_heads)
    : ln1(input_dim),
      mha(input_dim, n_heads),
      ln2(input_dim),
      ff(input_dim, hidden_dim),
      dropout1(0.1f),
      dropout2(0.1f) {}

Tensor TransformerBlock::forward(const Tensor& input, bool training) {
    Tensor norm_out1 = ln1.forward(input);
    Tensor attn_out = mha.forward(norm_out1);
    attn_out = dropout1.forward(attn_out, training);
    Tensor out1 = attn_out + input;  // Residual connection

    Tensor norm_out2 = ln2.forward(out1);
    Tensor ff_out = ff.forward(norm_out2);
    ff_out = dropout2.forward(ff_out, training);
    return ff_out + out1;           // Residual connection
}

Transformer::Transformer(int num_layers, int input_dim,
                        int hidden_dim, int n_heads) {
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(input_dim, hidden_dim, n_heads);
    }
}

Tensor Transformer::forward(const Tensor& input, bool training) {
    Tensor output = input;
    for (auto& block : blocks) {
        output = block.forward(output, training);
    }
    return output;
}