#include "transformer.hpp"

class TransformerBlock {
    Norm norm1, norm2;
    MLA mla;
    MoE moe;
    
public:
    TransformerBlock(int input_dim, int hidden_dim, int n_heads, int compress_dim)
        : mla(input_dim, hidden_dim, n_heads, compress_dim),
          moe(input_dim, hidden_dim, 4) {}  // 4 experts

    Tensor forward(const Tensor& input) {
        // First sub-layer (MLA)
        Tensor norm_out1 = norm1.forward(input);
        Tensor mla_out = mla.forward(norm_out1);
        Tensor out1 = mla_out.matmul(input);  // Residual connection
        
        // Second sub-layer (MoE)
        Tensor norm_out2 = norm2.forward(out1);
        Tensor moe_out = moe.forward(norm_out2);
        Tensor out2 = moe_out.matmul(out1);   // Residual connection
        
        return out2;
    }
};

Transformer::Transformer(int num_layers, int input_dim, int hidden_dim, 
                       int n_heads, int compress_dim) {
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