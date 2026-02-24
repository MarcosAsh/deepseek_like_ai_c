#include "layers/ad_lora.hpp"
#include <random>
#include <cmath>

ADLoRA::ADLoRA(int input_dim_, int output_dim_, int rank_, float alpha_)
    : input_dim(input_dim_), output_dim(output_dim_), rank(rank_), alpha(alpha_) {
    std::mt19937 gen(std::random_device{}());

    // Base weight W (frozen - not registered as parameter)
    Tensor tW(output_dim, input_dim);
    float r = std::sqrt(6.0f / (input_dim + output_dim));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto& v : tW.data) v = dist(gen);
    W = make_ad(tW);
    // W is NOT registered as parameter - it's frozen

    // LoRA A: down-projection, initialized with Kaiming uniform
    Tensor tA(rank, input_dim);
    float r_a = std::sqrt(6.0f / (input_dim + rank));
    std::uniform_real_distribution<float> dist_a(-r_a, r_a);
    for (auto& v : tA.data) v = dist_a(gen);
    A = make_ad(tA);
    register_parameter(A);

    // LoRA B: up-projection, initialized to zero (so initial output = W*x)
    Tensor tB(output_dim, rank);
    tB.fill(0.0f);
    B = make_ad(tB);
    register_parameter(B);

    // Bias
    Tensor tb(output_dim, 1);
    tb.fill(0.0f);
    bias = make_ad(tb);
    register_parameter(bias);
}

std::shared_ptr<ADTensor> ADLoRA::forward(const std::shared_ptr<ADTensor>& x) {
    // Base: W * x
    auto base = matmul(W, x);

    // LoRA: alpha * B @ A @ x
    auto Ax = matmul(A, x);       // [rank x seq_len]
    auto BAx = matmul(B, Ax);     // [output_dim x seq_len]
    float scale = alpha / static_cast<float>(rank);
    auto lora_out = scalar_mul(BAx, scale);

    // Combine: W*x + alpha/rank * B@A@x
    auto combined = add(base, lora_out);

    // Add bias
    int seq_len = x->val.cols;
    Tensor ones_row(1, seq_len);
    ones_row.data.assign(seq_len, 1.0f);
    auto ones = make_ad(ones_row);
    auto bias_broadcast = matmul(bias, ones);
    return add(combined, bias_broadcast);
}
