#include "layers/ad_multi_head_attention.hpp"
#include <random>
#include <stdexcept>
#include <cmath>
#include <limits>

ADMultiHeadAttention::ADMultiHeadAttention(int embed_dim_, int num_heads_, bool causal_)
    : embed_dim(embed_dim_), num_heads(num_heads_), causal(causal_) {
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim = embed_dim / num_heads;
    alibi_slopes.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        alibi_slopes[h] = std::pow(2.0f, -8.0f * static_cast<float>(h + 1) / static_cast<float>(num_heads));
    }
    Tensor tWq(embed_dim, embed_dim), tWk(embed_dim, embed_dim),
           tWv(embed_dim, embed_dim), tWo(embed_dim, embed_dim);
    std::mt19937 gen(std::random_device{}());
    float range = std::sqrt(6.0f / (2 * embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto &v : tWq.data) v = dist(gen);
    for (auto &v : tWk.data) v = dist(gen);
    for (auto &v : tWv.data) v = dist(gen);
    for (auto &v : tWo.data) v = dist(gen);
    W_q = make_ad(tWq); register_parameter(W_q);
    W_k = make_ad(tWk); register_parameter(W_k);
    W_v = make_ad(tWv); register_parameter(W_v);
    W_o = make_ad(tWo); register_parameter(W_o);
}

std::shared_ptr<ADTensor> ADMultiHeadAttention::forward(
    const std::shared_ptr<ADTensor>& input) {
    auto Q = matmul(W_q, input);
    auto K = matmul(W_k, input);
    auto V = matmul(W_v, input);
    int seq_len = input->val.cols;
    std::vector<std::shared_ptr<ADTensor>> heads;
    heads.reserve(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        auto Qh = slice(Q, offset, head_dim);
        auto Kh = slice(K, offset, head_dim);
        auto Vh = slice(V, offset, head_dim);
        auto Qh_T = transpose(Qh);
        auto scores = matmul(Qh_T, Kh);
        float scale = 1.0f / std::sqrt((float)head_dim);
        auto scores_scaled = scalar_mul(scores, scale);
        // ALiBi bias + causal mask
        {
            Tensor bias_t(seq_len, seq_len);
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    if (causal && j > i) {
                        bias_t.data[i * seq_len + j] = -std::numeric_limits<float>::infinity();
                    } else {
                        bias_t.data[i * seq_len + j] = -std::abs(j - i) * alibi_slopes[h];
                    }
                }
            }
            auto bias_ad = make_ad(bias_t);
            scores_scaled = add(scores_scaled, bias_ad);
        }
        // row-wise softmax
        Tensor row_max_t(seq_len, 1);
        for (int i = 0; i < seq_len; ++i) {
            float mx = scores_scaled->val.data[i * seq_len];
            for (int j = 1; j < seq_len; ++j) {
                mx = std::max(mx, scores_scaled->val.data[i * seq_len + j]);
            }
            row_max_t.data[i] = mx;
        }
        Tensor ones_row_t(1, seq_len);
        for (int j = 0; j < seq_len; ++j) ones_row_t.data[j] = 1.0f;
        auto ones_row = make_ad(ones_row_t);
        auto row_max_ad = make_ad(row_max_t);
        auto max_broadcast = matmul(row_max_ad, ones_row);
        auto scores_shifted = sub(scores_scaled, max_broadcast);
        auto scores_exp = exp_ad(scores_shifted);
        Tensor ones_col_t(seq_len, 1);
        for (int i = 0; i < seq_len; ++i) ones_col_t.data[i] = 1.0f;
        auto ones_col = make_ad(ones_col_t);
        auto denom_col = matmul(scores_exp, ones_col);
        auto denom = matmul(denom_col, ones_row);
        auto denom_recip = reciprocal(denom);
        auto attn = mul(scores_exp, denom_recip);
        auto attn_T = transpose(attn);
        auto head_out = matmul(Vh, attn_T);
        heads.push_back(head_out);
    }
    auto concat_out = concat(heads);
    auto out = matmul(W_o, concat_out);
    return out;
}