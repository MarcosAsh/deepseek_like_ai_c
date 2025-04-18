#include "layers/ad_linear.hpp"
#include <random>
#include <cmath>

ADLinear::ADLinear(int input_dim, int output_dim) {
    // Xavier initialization for W: [-r, r]
    Tensor tW(output_dim, input_dim);
    Tensor tb(output_dim, 1);
    std::mt19937 gen(std::random_device{}());
    float r = std::sqrt(6.0f / (input_dim + output_dim));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto &v : tW.data) v = dist(gen);
    // initialize bias to zero
    tb.fill(0.0f);
    // wrap in ADTensor and register
    W = make_ad(tW);
    register_parameter(W);
    b = make_ad(tb);
    register_parameter(b);
}

std::shared_ptr<ADTensor> ADLinear::forward(
    const std::shared_ptr<ADTensor>& x) const {
    // W * x
    auto y = matmul(W, x);
    // broadcast b across seq_len
    int seq_len = x->val.cols;
    Tensor ones_row_t(1, seq_len);
    ones_row_t.data.assign(seq_len, 1.0f);
    auto ones_row = make_ad(ones_row_t);
    auto b_mat = matmul(b, ones_row);  // [output_dim x seq_len]
    auto out = add(y, b_mat);
    return out;
}