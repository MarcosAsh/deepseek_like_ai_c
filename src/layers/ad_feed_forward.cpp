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
    int seq_len = x->val.cols;
    // Cache ones tensors for bias broadcast
    if (seq_len != cached_seq_len) {
        cached_ones1 = Tensor(1, seq_len);
        cached_ones1.data.assign(seq_len, 1.0f);
        cached_ones2 = Tensor(1, seq_len);
        cached_ones2.data.assign(seq_len, 1.0f);
        cached_seq_len = seq_len;
    }
    // First linear + bias
    auto lin1 = matmul(W1, x);
    auto ones_row1 = make_ad(cached_ones1);
    auto b1_mat = matmul(b1, ones_row1);
    auto h1 = add(lin1, b1_mat);
    // GELU activation: x * 0.5 * (1 + tanh(0.79788456*(x + 0.044715*x^3)))
    auto x3 = mul(mul(h1, h1), h1);
    auto inner = add(h1, scalar_mul(x3, 0.044715f));
    auto tanh_in = scalar_mul(inner, 0.79788456f);
    auto tanh_out = tanh_ad(tanh_in);
    auto one = make_ad(Tensor(h1->val.rows, h1->val.cols));
    one->val.fill(1.0f);
    auto gelu = mul(scalar_mul(h1, 0.5f), add(one, tanh_out));
    // Second linear + bias
    auto lin2 = matmul(W2, gelu);
    auto ones_row2 = make_ad(cached_ones2);
    auto b2_mat = matmul(b2, ones_row2);
    return add(lin2, b2_mat);
}