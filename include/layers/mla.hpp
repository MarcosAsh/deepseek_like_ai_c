#pragma once
#include "tensor.hpp"

class MLA {
public:
    MLA(int input_dim, int hidden_dim, int n_heads, int compress_dim);
    Tensor forward(const Tensor& input);

private:
    int d_in, d_hidden, n_heads, d_compress;

    Tensor W_dkv; // [d_compress x d_in]
    Tensor W_uk;  // [d_hidden x d_compress]
    Tensor W_uv;  // [d_hidden x d_compress]
    Tensor W_q;   // [d_hidden x d_in]
    Tensor W_o;   // [d_in x d_hidden]
};
