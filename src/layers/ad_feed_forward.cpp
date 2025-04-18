#include "layers/ad_feed_forward.hpp"
#include <random>

ADFeedForward::ADFeedForward(int embed_dim, int hidden_dim) {
    // Initialize parameters similarly to Linear layers
    Tensor tW1(hidden_dim, embed_dim), tb1(hidden_dim, 1);
    Tensor tW2(embed_dim, hidden_dim), tb2(embed_dim, 1);
    std::mt19937 gen(std::random_device{}());
    float range1 = std::sqrt(6.0f / (embed_dim + hidden_dim));
    std::uniform_real_distribution<float> dist1(-range1, range1);
    for (auto &v : tW1.data) v = dist1(gen);
    tb1.fill(0.0f);
    float range2 = std::sqrt(6.0f / (hidden_dim + embed_dim));
    std::uniform_real_distribution<float> dist2(-range2, range2);
    for (auto &v : tW2.data) v = dist2(gen);
    tb2.fill(0.0f);

    W1 = make_ad(tW1); register_parameter(W1);
    b1 = make_ad(tb1); register_parameter(b1);
    W2 = make_ad(tW2); register_parameter(W2);
    b2 = make_ad(tb2); register_parameter(b2);
}

std::shared_ptr<ADTensor> ADFeedForward::forward(const std::shared_ptr<ADTensor>& x) {
    // First linear + bias: h1 = W1 * x + b1
    // First linear + bias (broadcast b1 across seq_len)
    auto lin1 = matmul(W1, x);
    // create ones row [1 x seq_len]
    Tensor ones_row1_t(1, x->val.cols);
    ones_row1_t.data.assign(x->val.cols, 1.0f);
    auto ones_row1 = make_ad(ones_row1_t);
    auto b1_mat = matmul(b1, ones_row1);  // [hidden_dim x seq_len]
    auto h1 = add(lin1, b1_mat);
    // GELU activation: approximate
    // x * 0.5 * (1 + tanh(0.79788456*(x + 0.044715*x^3)))
    auto x3 = mul(mul(h1, h1), h1);
    auto inner = add(h1, scalar_mul(x3, 0.044715f));
    auto tanh_in = scalar_mul(inner, 0.79788456f);
    auto tanh_out = tanh_ad(tanh_in);
    auto one = make_ad(Tensor(h1->val.rows, h1->val.cols));
    one->val.fill(1.0f);
    auto gelu = mul(scalar_mul(h1, 0.5f), add(one, tanh_out));
    // Second linear + bias
    // Second linear + bias (broadcast b2)
    auto lin2 = matmul(W2, gelu);
    Tensor ones_row2_t(1, x->val.cols);
    ones_row2_t.data.assign(x->val.cols, 1.0f);
    auto ones_row2 = make_ad(ones_row2_t);
    auto b2_mat = matmul(b2, ones_row2);  // [embed_dim x seq_len]
    auto h2 = add(lin2, b2_mat);
    return h2;
}