#pragma once
#include "tensor.hpp"
#include "layers/mla.hpp"
#include "layers/moe.hpp"
#include "layers/norm.hpp"
#include <vector>

class Transformer {
public:
    Transformer(int num_layers, int input_dim, int hidden_dim, int n_heads, int compress_dim);
    Tensor forward(const Tensor& input);

private:
    std::vector<class TransformerBlock> blocks;
};