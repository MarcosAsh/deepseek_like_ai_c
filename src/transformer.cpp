#include "transformer.hpp"

TransformerBlock::TransformerBlock(int input_dim, int hidden_dim, 
                                 int n_heads, int compress_dim)
    : mla(input_dim, hidden_dim, n_heads, compress_dim),
      moe(input_dim, hidden_dim, 4) {} // Using 4 experts

Tensor TransformerBlock::forward(const Tensor& input) {
    Tensor norm_out1 = norm1.forward(input);
    Tensor mla_out = mla.forward(norm_out1);
    Tensor out1 = mla_out + input;  // Residual connection
    
    Tensor norm_out2 = norm2.forward(out1);
    Tensor moe_out = moe.forward(norm_out2);
    return moe_out + out1;         // Residual connection
}

Transformer::Transformer(int num_layers, int input_dim, 
                        int hidden_dim, int n_heads, int compress_dim) {
    for (int i = 0; i < num_layers; ++i) {
        blocks.emplace_back(input_dim, hidden_dim, n_heads, compress_dim);
    }
}

Tensor Transformer::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& block : blocks) {
        output = block.forward(output);
    }
    return output;
}